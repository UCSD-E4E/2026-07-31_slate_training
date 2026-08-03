"""Ground-truth-free correctness check: is the recovered board plane right?

The laser is a **fixed ray in camera space**. So for a dive, intersecting each
frame's labelled laser pixel with that frame's estimated board plane must yield
3-D points that are *collinear*. Correct planes -> tight line; wrong planes ->
scatter.

This needs **no slate labels at all**, which is what makes it usable on the pool
dives (65, 71), where there are none. It is also a stronger check than
cross-frame pose consensus: consensus asks whether the per-frame poses agree
with each other, and so passes when every fit is wrong in the same way.
Collinearity tests them against an external physical constraint instead.

Reef baseline, measured earlier from *human* labels: per-dive residuals of
1.4-63 mm off the best-fit line.

    nix develop
    uv run --extra mask python scripts/laser_collinearity.py --dives 65,71
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pymupdf
import rawpy

HERE = os.path.dirname(os.path.abspath(__file__))
NAS = os.path.expanduser("~/mnt/fishsense_data/REEF/data")
CHECKPOINT = os.path.join(HERE, "..", "data", "seg", "board_unet.pt")

_TPL: dict = {}


def template(slate_path, dpi):
    """Binarized template render, cached per slate."""
    if slate_path not in _TPL:
        with pymupdf.open(os.path.join(NAS, slate_path)) as doc:
            pix = doc.load_page(0).get_pixmap(dpi=dpi)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, binar = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        _TPL[slate_path] = binar
    return _TPL[slate_path]


def line_residual_mm(points):
    """RMS distance of 3-D points from their best-fit line, in mm."""
    P = np.asarray(points, dtype=float)
    centred = P - P.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    # Total variance off the principal direction.
    return float(np.sqrt((singular[1] ** 2 + singular[2] ** 2) / len(P))) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dives", default="65,71")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    wanted = {int(x) for x in args.dives.split(",")}

    from fishsense_core.slate import estimate_plane
    from fishsense_core.plane import Plane

    masker = None
    if os.path.exists(CHECKPOINT):
        from fishsense_core.slate import BoardMasker
        masker = BoardMasker.from_checkpoint(CHECKPOINT)

    rows = json.load(open(os.path.join(HERE, "..", "data", "pool_frames.json"),
                          encoding="utf-8"))
    rows = [r for r in rows if r["dive_id"] in wanted and r["laser_x"] is not None]
    if args.limit:
        rows = rows[: args.limit]

    per_dive: dict = {}
    for n, r in enumerate(rows, 1):
        path = os.path.join(NAS, r["path"])
        if not os.path.exists(path):
            continue
        K = np.array(r["camera_matrix"], float)
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        bgr = cv2.undistort(np.ascontiguousarray(rgb[:, :, ::-1]), K,
                            np.array(r["dist"], float))

        mask = masker.predict(bgr) if masker is not None else None
        est = estimate_plane(bgr, template(r["slate_path"], r["dpi"]),
                             r["template_points"], r["dpi"], K, board_mask=mask)
        print(f"\r{n}/{len(rows)}", end="", flush=True)
        if est is None:
            continue

        # Intersect the laser ray with the estimated plane.
        plane = est.plane
        k_inv = np.linalg.inv(K)
        ray = k_inv @ np.array([r["laser_x"], r["laser_y"], 1.0])
        denom = float(plane.normal @ ray)
        if abs(denom) < 1e-12:
            continue
        point = ray * (plane.distance / denom)
        if not np.isfinite(point).all():
            continue
        per_dive.setdefault(r["dive_id"], []).append(
            {"ecc": float(est.ecc_score), "point": point.tolist(),
             "image_id": r["image_id"]}
        )
    print()

    out = os.path.join(HERE, "..", "data", "pool_collinearity.json")
    json.dump(per_dive, open(out, "w"), indent=1)

    print(f"\n{'dive':>6}{'n':>5}{'gated n':>9}{'med ECC':>10}"
          f"{'line resid (all)':>19}{'line resid (gated)':>21}")
    print("-" * 72)
    for dive, items in sorted(per_dive.items()):
        eccs = np.array([x["ecc"] for x in items])
        pts = np.array([x["point"] for x in items])
        gated = eccs >= 0.80
        all_r = line_residual_mm(pts) if len(pts) >= 3 else float("nan")
        gat_r = line_residual_mm(pts[gated]) if gated.sum() >= 3 else float("nan")
        print(f"{dive:>6}{len(items):>5}{int(gated.sum()):>9}{np.median(eccs):>10.3f}"
              f"{all_r:>17.1f}mm{gat_r:>19.1f}mm")
    print("\nreef baseline from human labels: 1.4-63 mm")


if __name__ == "__main__":
    main()
