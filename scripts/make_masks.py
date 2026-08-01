"""Build a board-segmentation dataset from the existing human labels.

No new annotation needed. Each `DiveSlateLabel` gives point correspondences
between the template render and the photo, so a homography fitted to them warps
the template's page outline into the image — that outline *is* the board. 104
labeled frames therefore yield 104 exact board masks for free.

Why segmentation and not keypoints: the classical estimator's remaining
failures are localization, not accuracy. Gated offsets are already 27-58 mm,
but the template search has to scan the whole frame, so reef texture competes
with the board and every extra hypothesis raises the noise floor (measured: a
finer scale grid made things *worse*, 67% -> 63%). A mask that says "the board
is here" removes the competition entirely and leaves all the existing geometry
untouched.

    uv run python scripts/make_masks.py --size 512
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pymupdf
import rawpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from slate_training.contracts import (  # noqa: E402
    composite_panel_width,
    drop_skipped,
    repair_panel_offset,
)

HERE = os.path.dirname(os.path.abspath(__file__))
NAS = os.path.expanduser("~/mnt/fishsense_data/REEF/data")
OUT = os.path.join(HERE, "..", "data", "seg")
PHOTO_W, PHOTO_H = 4014, 3016
_TPL: dict = {}


def template(name, path, dpi):
    if name not in _TPL:
        with pymupdf.open(os.path.join(NAS, path)) as doc:
            page = doc.load_page(0)
            rect = page.rect
            pix = page.get_pixmap(dpi=dpi)
        _TPL[name] = (pix.width, pix.height,
                      composite_panel_width(rect.width / rect.height, PHOTO_H))
    return _TPL[name]


def board_polygon(row, panel, tpl_w, tpl_h):
    """Warp the template page outline into the image via the label homography.

    Returns the board's 4 image-space corners, or None when the fit is
    degenerate (too few visible points, or a homography that collapses).
    """
    tpl_pts = [tuple(p) for p in row["template_points"]]
    skipped = row["skipped"] if isinstance(row["skipped"], list) else []
    src = np.array(drop_skipped(tpl_pts, skipped), dtype=np.float32)
    dst = np.array(
        repair_panel_offset([tuple(p) for p in row["label_points"]], panel, PHOTO_W),
        dtype=np.float32,
    )
    if len(src) != len(dst) or len(src) < 4:
        return None

    homography, _ = cv2.findHomography(src, dst, method=0)
    if homography is None:
        return None

    corners = np.array(
        [[0, 0], [tpl_w, 0], [tpl_w, tpl_h], [0, tpl_h]], dtype=np.float32
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, homography).reshape(4, 2)
    if not np.isfinite(warped).all():
        return None
    # Reject fits that place the board absurdly (a sign of a bad homography).
    if cv2.contourArea(warped.astype(np.float32)) < 1000:
        return None
    return warped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512, help="output width")
    args = ap.parse_args()

    rows = json.load(open(os.path.join(HERE, "..", "data", "slate_geometry_export.json"),
                          encoding="utf-8"))
    meta = json.load(open(os.path.join(HERE, "..", "data", "image_paths.json"),
                          encoding="utf-8"))
    slate_paths = json.load(open(os.path.join(HERE, "..", "data", "slate_paths.json"),
                                 encoding="utf-8"))
    templates = {t["name"]: t for t in json.load(
        open(os.path.join(HERE, "..", "data", "slate_templates.json"), encoding="utf-8"))}

    seen, uniq = set(), []
    for r in rows:
        if r["label_id"] in seen:
            continue
        seen.add(r["label_id"])
        uniq.append(r)

    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)

    width = args.size
    height = int(round(width * PHOTO_H / PHOTO_W))
    index, skipped_count = [], 0

    for i, r in enumerate(uniq, 1):
        name = r["slate_name"]
        tpl_w, tpl_h, panel = template(name, slate_paths[name], templates[name]["dpi"])
        poly = board_polygon(r, panel, tpl_w, tpl_h)
        if poly is None:
            skipped_count += 1
            continue

        with rawpy.imread(os.path.join(NAS, meta["paths"][str(r["image_id"])])) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        bgr = cv2.undistort(
            np.ascontiguousarray(rgb[:, :, ::-1]),
            np.array(r["camera_matrix"], float),
            np.array(meta["dist"][str(r["image_id"])], float),
        )

        mask = np.zeros(bgr.shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)

        small_img = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        stem = f"{r['label_id']}"
        cv2.imwrite(os.path.join(OUT, "images", stem + ".png"), small_img)
        cv2.imwrite(os.path.join(OUT, "masks", stem + ".png"), small_mask)
        index.append({
            "label_id": r["label_id"], "dive": r["dive_id"], "slate": name,
            "mask_frac": float((small_mask > 0).mean()),
        })
        print(f"\r{i}/{len(uniq)}", end="", flush=True)
    print()

    json.dump({"width": width, "height": height, "items": index},
              open(os.path.join(OUT, "index.json"), "w"), indent=1)

    fracs = np.array([x["mask_frac"] for x in index])
    print(f"\nwrote {len(index)} pairs ({skipped_count} skipped) at {width}x{height}")
    print(f"board covers median {100*np.median(fracs):.2f}% of frame "
          f"(min {100*fracs.min():.2f}%, max {100*fracs.max():.2f}%)")
    dives = {}
    for x in index:
        dives[x["dive"]] = dives.get(x["dive"], 0) + 1
    print("per-dive:", dict(sorted(dives.items())))


if __name__ == "__main__":
    main()
