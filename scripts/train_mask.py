"""Train a small UNet to segment the slate board.

Purpose is narrow and deliberate: **restrict where the template search looks.**
The classical estimator's geometry is already good (gated plane offsets 27-58mm,
5.7px points) — what fails is localization, because the search must scan the
whole frame and reef texture competes with the board. Adding search hypotheses
makes that worse, not better (measured: a finer scale grid dropped coverage
67% -> 63%), because every extra candidate is another chance for background to
win. A mask removes the competition instead of out-tuning it.

So the bar is low and specific: the mask only has to say roughly *where* the
board is. All geometry downstream stays classical and untouched.

Two properties of this dataset drive the design:

* **Severe class imbalance** — the board covers a median 1.1% of frame
  (min 0.33%). Plain BCE would happily predict all-background at 99% accuracy,
  so the loss is Dice + BCE.
* **104 frames from 8 dives.** The split is by *dive*, never by frame: frames
  within a dive share board, camera, water and lighting, so a frame-level split
  leaks badly and would overstate accuracy.

    LD_LIBRARY_PATH=/run/opengl-driver/lib uv run --extra train \\
        python scripts/train_mask.py --epochs 60
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
SEG = os.path.join(HERE, "..", "data", "seg")
# Held out entirely from training. Chosen to span three slate families so the
# validation number is not just "does it work on V-Slate 1".
VAL_DIVES = {349, 383, 279}


class BoardData(Dataset):
    def __init__(self, items, size, train):
        self.items, self.size, self.train = items, size, train

    def __len__(self):
        return len(self.items)

    def _augment(self, img, mask):
        """Flips, 90-degree rotations, scale/shift, and photometric jitter.

        Underwater colour varies hugely between sites, and with 8 dives the
        model would otherwise memorise each dive's colour cast rather than the
        board's appearance.
        """
        if np.random.rand() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if np.random.rand() < 0.5:
            img, mask = img[::-1], mask[::-1]
        # No np.rot90: the frame is not square, so k=1/3 transposes it and the
        # batch fails to stack. Arbitrary rotation comes from the affine warp
        # below, which preserves shape — and boards genuinely appear at any
        # angle (one failure was a chevron held at ~225 degrees).
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)

        if np.random.rand() < 0.9:
            h, w = mask.shape
            scale = np.random.uniform(0.75, 1.35)
            angle = np.random.uniform(-180, 180)
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            matrix[0, 2] += np.random.uniform(-0.08, 0.08) * w
            matrix[1, 2] += np.random.uniform(-0.08, 0.08) * h
            img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)

        img = img.astype(np.float32)
        img *= np.random.uniform(0.75, 1.3)
        img += np.random.uniform(-25, 25)
        img *= np.random.uniform(0.85, 1.15, size=(1, 1, 3))  # per-channel cast
        return np.clip(img, 0, 255), mask

    def __getitem__(self, i):
        item = self.items[i]
        img = cv2.imread(os.path.join(SEG, "images", f"{item['label_id']}.png"))
        mask = cv2.imread(os.path.join(SEG, "masks", f"{item['label_id']}.png"), 0)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        if self.train:
            img, mask = self._augment(img, mask)
        img = np.asarray(img, np.float32) / 255.0
        img = (img - 0.45) / 0.25
        return (
            torch.from_numpy(img.transpose(2, 0, 1).copy()),
            torch.from_numpy((mask > 127).astype(np.float32)[None].copy()),
        )


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Deliberately small (~1.9M params). 104 training frames cannot support a
    large model, and the task — locate a high-contrast rectangle — is easy."""

    def __init__(self, ch=(24, 48, 96, 192)):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.d1, self.d2, self.d3 = block(3, c1), block(c1, c2), block(c2, c3)
        self.bottom = block(c3, c4)
        self.u3, self.c3 = nn.ConvTranspose2d(c4, c3, 2, 2), block(c3 * 2, c3)
        self.u2, self.c2 = nn.ConvTranspose2d(c3, c2, 2, 2), block(c2 * 2, c2)
        self.u1, self.c1 = nn.ConvTranspose2d(c2, c1, 2, 2), block(c1 * 2, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(F.max_pool2d(d1, 2))
        d3 = self.d3(F.max_pool2d(d2, 2))
        b = self.bottom(F.max_pool2d(d3, 2))
        x = self.c3(torch.cat([self.u3(b), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))
        return self.head(x)


def dice_bce(logits, target, eps=1.0):
    """Dice carries the gradient signal; BCE stabilises early training.

    Dice is essential here: at ~1% positive pixels, BCE alone is minimised by
    predicting all-background.
    """
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum((1, 2, 3)) + eps
    den = probs.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return (1 - num / den).mean() + F.binary_cross_entropy_with_logits(logits, target)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """IoU plus the metric that actually matters: does the predicted blob's
    centroid land inside the true board?"""
    model.eval()
    ious, hits, n = [], 0, 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        pred = (torch.sigmoid(model(img)) > threshold).float()
        inter = (pred * mask).sum((1, 2, 3))
        union = ((pred + mask) > 0).float().sum((1, 2, 3))
        ious += (inter / union.clamp(min=1)).tolist()
        for p, m in zip(pred.cpu().numpy()[:, 0], mask.cpu().numpy()[:, 0]):
            n += 1
            # Largest connected component, not the centroid of all positives:
            # with several blobs the global centroid lands between them, which
            # both understates the model and misrepresents how it is used
            # downstream (the search gets one region to look in).
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(
                p.astype(np.uint8), 8)
            if num <= 1:
                continue
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            cx, cy = centroids[biggest]
            cy = int(np.clip(cy, 0, m.shape[0] - 1))
            cx = int(np.clip(cx, 0, m.shape[1] - 1))
            if m[cy, cx] > 0:
                hits += 1
    return float(np.mean(ious)), hits / max(n, 1)


@torch.no_grad()
def write_oof(model, items, size, device, out_dir):
    """Save this fold's predicted masks for its held-out frames.

    Out-of-fold predictions are the only honest way to evaluate the integrated
    pipeline: every frame's mask comes from a model that never saw that frame's
    dive, so downstream coverage numbers carry no leakage.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    loader = DataLoader(BoardData(items, size, False), batch_size=4, num_workers=2)
    i = 0
    for img, _ in loader:
        prob = torch.sigmoid(model(img.to(device))).cpu().numpy()[:, 0]
        for p in prob:
            cv2.imwrite(os.path.join(out_dir, f"{items[i]['label_id']}.png"),
                        (p * 255).astype(np.uint8))
            i += 1


def run_fold(items, val_dives, size, args, device, save_path=None, verbose=True,
             oof_dir=None):
    """Train one fold and return (final_metrics, best_metrics).

    Both are reported deliberately. `best` early-stops on the held-out fold,
    which selects on the test set and is therefore **optimistic** — it is an
    upper bound, not an estimate. `final` uses a fixed epoch budget and is the
    honest number.
    """
    train_items = [x for x in items if x["dive"] not in val_dives]
    val_items = [x for x in items if x["dive"] in val_dives]
    if not train_items:
        return None, None
    # An empty holdout is legitimate: it means "train on everything" to produce
    # the deployable checkpoint. Honest accuracy comes from the leave-one-dive-
    # out run; this model exists to be shipped, not measured, so it reports
    # train-set metrics only and always saves.
    deploy = not val_items
    if deploy:
        val_items = train_items

    torch.manual_seed(0)
    model = UNet().to(device)
    train_loader = DataLoader(BoardData(train_items, size, True), batch_size=args.batch,
                              shuffle=True, num_workers=4, drop_last=len(train_items) > args.batch)
    val_loader = DataLoader(BoardData(val_items, size, False), batch_size=args.batch,
                            num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(train_loader)))
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    best = (0.0, 0.0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for img, mask in train_loader:
            img, mask = img.to(device, non_blocking=True), mask.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                loss = dice_bce(model(img), mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            iou, hit = evaluate(model, val_loader, device)
            if deploy or hit > best[1]:
                best = (iou, hit)
                if save_path:
                    torch.save({"model": model.state_dict(), "size": size}, save_path)
            if verbose and epoch % 40 == 0:
                print(f"    epoch {epoch:3d}  IoU {iou:.3f}  hit {hit:.2f}")
    final = evaluate(model, val_loader, device)
    if oof_dir:
        write_oof(model, val_items, size, device, oof_dir)
    return final, best


def cross_validate(items, size, args, device):
    """Leave-one-dive-out over all 8 dives."""
    dives = sorted({x["dive"] for x in items})
    print(f"leave-one-dive-out over {len(dives)} dives, {args.epochs} epochs each\n")
    rows = []
    for dive in dives:
        n_val = sum(1 for x in items if x["dive"] == dive)
        slates = {x["slate"] for x in items if x["dive"] == dive}
        print(f"  fold dive={dive} ({n_val} frames, {'/'.join(sorted(slates))})")
        final, best = run_fold(items, {dive}, size, args, device, verbose=False,
                               oof_dir=(os.path.join(SEG, "oof") if args.save_oof else None))
        if final is None:
            continue
        rows.append({"dive": dive, "n": n_val, "slate": "/".join(sorted(slates)),
                     "final_iou": final[0], "final_hit": final[1],
                     "best_iou": best[0], "best_hit": best[1]})
        print(f"     final IoU {final[0]:.3f} hit {final[1]:.2f}"
              f"   | best(optimistic) IoU {best[0]:.3f} hit {best[1]:.2f}")

    print(f"\n{'dive':>6}{'n':>5}{'slate':>16}{'final hit':>12}{'best hit':>11}")
    print("-" * 50)
    for r in rows:
        print(f"{r['dive']:>6}{r['n']:>5}{r['slate']:>16}{r['final_hit']:>12.2f}{r['best_hit']:>11.2f}")
    fh = np.array([r["final_hit"] for r in rows])
    bh = np.array([r["best_hit"] for r in rows])
    w = np.array([r["n"] for r in rows], dtype=float)
    print("-" * 50)
    print(f"{'mean':>6}{int(w.sum()):>5}{'':>16}{fh.mean():>12.2f}{bh.mean():>11.2f}")
    print(f"{'wtd':>6}{'':>5}{'':>16}{float((fh*w).sum()/w.sum()):>12.2f}"
          f"{float((bh*w).sum()/w.sum()):>11.2f}")
    print("\nfinal = fixed epoch budget (honest). "
          "best = early-stopped on the held-out fold (optimistic upper bound).")
    json.dump(rows, open(os.path.join(SEG, "cv_results.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--cv", action="store_true", help="leave-one-dive-out")
    ap.add_argument("--save-oof", action="store_true",
                    help="write out-of-fold mask predictions for every frame")
    args = ap.parse_args()

    index = json.load(open(os.path.join(SEG, "index.json"), encoding="utf-8"))
    items = index["items"]
    size = (args.width, int(round(args.width * 384 / 512)) // 32 * 32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cv:
        cross_validate(items, size, args, device)
        return

    train_items = [x for x in items if x["dive"] not in VAL_DIVES]
    val_items = [x for x in items if x["dive"] in VAL_DIVES]
    print(f"train {len(train_items)} frames / {len({x['dive'] for x in train_items})} dives"
          f"  |  val {len(val_items)} frames / {len(VAL_DIVES)} dives (held out)")
    print(f"input {size[0]}x{size[1]}")

    torch.manual_seed(0)
    model = UNet().to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M  device {device}")

    train_loader = DataLoader(BoardData(train_items, size, True), batch_size=args.batch,
                              shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(BoardData(val_items, size, False), batch_size=args.batch,
                            num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(train_loader)))
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device, non_blocking=True), mask.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                loss = dice_bce(model(img), mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        if epoch % 5 == 0 or epoch == args.epochs:
            iou, hit = evaluate(model, val_loader, device)
            flag = ""
            if hit > best:
                best = hit
                torch.save({"model": model.state_dict(), "size": size},
                           os.path.join(SEG, "board_unet.pt"))
                flag = "  *saved"
            print(f"epoch {epoch:3d}  loss {total/max(1,len(train_loader)):.4f}"
                  f"  val IoU {iou:.3f}  centroid-hit {hit:.2f}{flag}")

    print(f"\nbest held-out centroid-hit rate: {best:.2f}")


if __name__ == "__main__":
    main()
