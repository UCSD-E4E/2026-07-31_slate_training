"""Crop the board region from real frames using the human labels as a guide.

Purely diagnostic: shows the baseline what it has to segment (contrast, glare,
turbidity, board-vs-water separation) at the scale it actually occurs, before
any detector is written.

    uv run python scripts/crop_boards.py --n 6
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
    repair_panel_offset,
)

HERE = os.path.dirname(os.path.abspath(__file__))
NAS = os.path.expanduser("~/mnt/fishsense_data/REEF/data")
OUT = os.path.join(HERE, "..", "data", "crops")
PHOTO_W = 4014
MARGIN = 1.6  # pad the label bbox by this factor to include board + water


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    rows = json.load(open(os.path.join(HERE, "..", "data", "slate_geometry_export.json"),
                          encoding="utf-8"))
    meta = json.load(open(os.path.join(HERE, "..", "data", "image_paths.json"),
                          encoding="utf-8"))
    slate_paths = json.load(open(os.path.join(HERE, "..", "data", "slate_paths.json"),
                                 encoding="utf-8"))

    seen, uniq = set(), []
    for r in rows:
        if r["label_id"] in seen:
            continue
        seen.add(r["label_id"])
        uniq.append(r)

    by_slate = {}
    for r in uniq:
        by_slate.setdefault(r["slate_name"], []).append(r)
    picks = [v[0] for v in by_slate.values()]
    for v in by_slate.values():
        picks.extend(v[1:3])
    picks = picks[: args.n]

    os.makedirs(OUT, exist_ok=True)
    panels = {}

    for r in picks:
        name = r["slate_name"]
        if name not in panels:
            with pymupdf.open(os.path.join(NAS, slate_paths[name])) as doc:
                rect = doc.load_page(0).rect
            panels[name] = composite_panel_width(rect.width / rect.height, 3016)

        orf = os.path.join(NAS, meta["paths"][str(r["image_id"])])
        with rawpy.imread(orf) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        img = cv2.undistort(
            bgr,
            np.array(r["camera_matrix"], float),
            np.array(meta["dist"][str(r["image_id"])], float),
        )

        pts = np.array(
            repair_panel_offset(
                [tuple(p) for p in r["label_points"]], panels[name], PHOTO_W
            ),
            dtype=float,
        )
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        half = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * MARGIN / 2 + 40
        x0, x1 = int(max(0, cx - half)), int(min(img.shape[1], cx + half))
        y0, y1 = int(max(0, cy - half)), int(min(img.shape[0], cy + half))

        crop = img[y0:y1, x0:x1]
        out = os.path.join(OUT, f"crop_{r['label_id']}_{name.replace(' ', '_')}.png")
        cv2.imwrite(out, crop)
        print(f"{name:<15} label={r['label_id']:<5} board~{np.ptp(pts[:,0]):.0f}x"
              f"{np.ptp(pts[:,1]):.0f}px  crop={crop.shape[1]}x{crop.shape[0]}")


if __name__ == "__main__":
    main()
