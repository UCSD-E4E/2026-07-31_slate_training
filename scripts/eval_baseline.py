"""Evaluate the classical baseline against the human-derived plane oracle.

Reports the only metric that reaches the calibration: agreement of the board
plane, as (normal angle in degrees, offset in mm). Per-point pixel error is
deliberately not reported -- the points are an arbitrary parameterization.

    uv run python scripts/eval_baseline.py [--n N]
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
from fishsense_core.slate import estimate_plane  # noqa: E402
from slate_training.contracts import (  # noqa: E402
    composite_panel_width,
    drop_skipped,
    repair_panel_offset,
)
from fishsense_core.plane import plane_difference, plane_from_pose  # noqa: E402

INCH_TO_M = 0.0254
PHOTO_W = 4014
HERE = os.path.dirname(os.path.abspath(__file__))
NAS = os.path.expanduser("~/mnt/fishsense_data/REEF/data")

_TPL: dict = {}


def template(name, path, dpi):
    """Binarized render + panel width, cached per slate."""
    if name not in _TPL:
        with pymupdf.open(os.path.join(NAS, path)) as doc:
            page = doc.load_page(0)
            rect = page.rect
            pix = page.get_pixmap(dpi=dpi)
            raw = np.frombuffer(pix.samples, dtype=np.uint8)
            rgb = raw.reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, binar = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        _TPL[name] = (binar, composite_panel_width(rect.width / rect.height, 3016))
    return _TPL[name]


def human_plane(row, panel, K):
    """Oracle plane from the (repaired) human labeling."""
    tpl = [tuple(p) for p in row["template_points"]]
    skipped = row["skipped"] if isinstance(row["skipped"], list) else []
    src = drop_skipped(tpl, skipped)
    body = np.zeros((len(src), 3), np.float32)
    body[:, :2] = (np.array(src) / float(row["dpi"])) * INCH_TO_M
    img = np.array(
        repair_panel_offset(
            [tuple(p) for p in row["label_points"]], panel, PHOTO_W
        ),
        dtype=float,
    )
    ok, rvec, tvec = cv2.solvePnP(body, img, K, np.zeros((5,)))
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    return plane_from_pose(rot, tvec.reshape(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--blur", type=float, default=0.0)
    ap.add_argument("--oof-masks", action="store_true",
                    help="use out-of-fold learned board masks as extra candidates")
    ap.add_argument("--stride", type=int, default=1, help="subsample corpus")
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
    if args.stride > 1:
        uniq = uniq[:: args.stride]
    if args.n:
        uniq = uniq[: args.n]

    results, failures = [], []
    for i, r in enumerate(uniq, 1):
        name = r["slate_name"]
        tpl_meta = templates[name]
        tpl_gray, panel = template(name, slate_paths[name], tpl_meta["dpi"])
        K = np.array(r["camera_matrix"], float)

        orf = os.path.join(NAS, meta["paths"][str(r["image_id"])])
        with rawpy.imread(orf) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        bgr = cv2.undistort(
            np.ascontiguousarray(rgb[:, :, ::-1]), K,
            np.array(meta["dist"][str(r["image_id"])], float),
        )

        truth = human_plane(r, panel, K)
        truth_pts = np.array(
            repair_panel_offset([tuple(q) for q in r["label_points"]], panel, PHOTO_W),
            dtype=float,
        )
        board_mask = None
        if args.oof_masks:
            mask_path = os.path.join(HERE, "..", "data", "seg", "oof",
                                     f"{r['label_id']}.png")
            if os.path.exists(mask_path):
                board_mask = cv2.imread(mask_path, 0)
        est = estimate_plane(
            bgr, tpl_gray, tpl_meta["reference_points"], tpl_meta["dpi"], K,
            template_blur=args.blur, board_mask=board_mask,
        )
        if est is None or truth is None:
            failures.append((r["label_id"], name, "no_board" if est is None else "no_truth"))
        else:
            angle, mm = plane_difference(truth, est.plane)
            # Per-point pixel error: the wrong metric for calibration (the
            # points are an arbitrary parameterization) but exactly the right
            # one for a Label Studio pre-annotation, where the labeler's work
            # is dragging dots. Compare only the points the human placed.
            keep = [i for i in range(len(est.image_points))
                    if i not in set(r["skipped"] or [] if isinstance(r["skipped"], list) else [])]
            pred_pts = est.image_points[keep] if len(keep) == len(truth_pts) else None
            px = (float(np.median(np.linalg.norm(pred_pts - truth_pts, axis=1)))
                  if pred_pts is not None else float("nan"))
            results.append({
                "label_id": r["label_id"], "dive": r["dive_id"], "slate": name,
                "angle_deg": angle, "offset_mm": mm,
                "ecc": est.ecc_score, "rms": est.reprojection_rms, "px_err": px,
            })
        print(f"\r{i}/{len(uniq)}", end="", flush=True)
    print()

    out = os.path.join(HERE, "..", "data", "baseline_eval.json")
    json.dump({"results": results, "failures": failures}, open(out, "w"), indent=2)

    print(f"\n{len(results)} estimated, {len(failures)} failed "
          f"({100*len(failures)/max(1,len(uniq)):.0f}% failure)\n")
    if not results:
        return
    print(f"{'slate':<16}{'n':>4}{'med angle':>12}{'p90':>9}"
          f"{'med offset':>13}{'p90':>10}{'med ecc':>10}")
    print("-" * 76)
    by = {}
    for x in results:
        by.setdefault(x["slate"], []).append(x)
    for slate in sorted(by):
        v = by[slate]
        a = np.array([x["angle_deg"] for x in v])
        m = np.array([x["offset_mm"] for x in v])
        e = np.array([x["ecc"] for x in v])
        print(f"{slate:<16}{len(v):>4}{np.median(a):>10.2f}d{np.percentile(a,90):>8.2f}"
              f"{np.median(m):>11.1f}mm{np.percentile(m,90):>9.1f}{np.median(e):>10.3f}")
    a = np.array([x["angle_deg"] for x in results])
    m = np.array([x["offset_mm"] for x in results])
    print("-" * 76)
    print(f"{'ALL':<16}{len(results):>4}{np.median(a):>10.2f}d{np.percentile(a,90):>8.2f}"
          f"{np.median(m):>11.1f}mm{np.percentile(m,90):>9.1f}")


if __name__ == "__main__":
    main()
