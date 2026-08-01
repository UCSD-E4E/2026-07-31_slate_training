"""Measure which image cue actually separates the board from the scene.

The first baseline failed because the board is photometrically identical to the
water at full-frame scale (grey 101 vs 102, saturation 165 vs 166). But the
board is locally obvious -- a black pattern on a light panel is a strong local
contrast signature that smooth water and fine reef texture lack.

Rather than guess again, this scores candidate cues by how well each separates
board pixels from the rest of the frame, using the human labels to define the
board region. Separation is reported as a d-prime-style score:

    (median_board - median_background) / MAD_background

so cues with different units are comparable. A cue needs to clear roughly 2-3
to be usable as a detector on its own.

    uv run python scripts/probe_cues.py --n 8
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
PHOTO_W = 4014


def gray_world(bgr):
    """Crude underwater colour correction: equalise per-channel means.

    Underwater the red channel is attenuated, leaving everything blue; if the
    board's whiteness is recoverable at all, this is the cheapest way to get it.
    """
    out = bgr.astype(np.float32)
    means = out.reshape(-1, 3).mean(axis=0)
    out *= means.mean() / np.maximum(means, 1e-6)
    return np.clip(out, 0, 255).astype(np.uint8)


def local_std(gray, ksize):
    """Standard deviation in a ksize box around each pixel."""
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (ksize, ksize))
    sq = cv2.boxFilter(f * f, -1, (ksize, ksize))
    return np.sqrt(np.maximum(sq - mean * mean, 0))


def local_range(gray, ksize):
    """Dilate minus erode: local max-min contrast."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.dilate(gray, k).astype(np.float32) - cv2.erode(gray, k).astype(np.float32)


def cues(bgr, ksize):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gw = gray_world(bgr)
    gw_hsv = cv2.cvtColor(gw, cv2.COLOR_BGR2HSV)
    sobel = np.hypot(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    return {
        "gray": gray.astype(np.float32),
        "saturation": cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32),
        "grayworld_V": gw_hsv[:, :, 2].astype(np.float32),
        "grayworld_lowS": 255.0 - gw_hsv[:, :, 1].astype(np.float32),
        "local_std": local_std(gray, ksize),
        "local_range": local_range(gray, ksize),
        "grad_density": cv2.boxFilter(sobel, -1, (ksize, ksize)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
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
    picks = []
    for v in by_slate.values():
        picks.extend(v[: max(1, args.n // len(by_slate))])
    picks = picks[: args.n]

    panels, scores = {}, {}
    for r in picks:
        name = r["slate_name"]
        if name not in panels:
            with pymupdf.open(os.path.join(NAS, slate_paths[name])) as doc:
                rect = doc.load_page(0).rect
            panels[name] = composite_panel_width(rect.width / rect.height, 3016)

        with rawpy.imread(os.path.join(NAS, meta["paths"][str(r["image_id"])])) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        bgr = cv2.undistort(
            np.ascontiguousarray(rgb[:, :, ::-1]),
            np.array(r["camera_matrix"], float),
            np.array(meta["dist"][str(r["image_id"])], float),
        )

        pts = np.array(
            repair_panel_offset([tuple(p) for p in r["label_points"]],
                                panels[name], PHOTO_W), dtype=float)
        x0, x1 = int(pts[:, 0].min()), int(pts[:, 0].max())
        y0, y1 = int(pts[:, 1].min()), int(pts[:, 1].max())
        # Window sized to the pattern -- that is the scale the cue lives at.
        ksize = int(max(15.0, (np.ptp(pts[:, 0]) + np.ptp(pts[:, 1])) / 8.0)) | 1

        board_mask = np.zeros(bgr.shape[:2], bool)
        board_mask[y0:y1, x0:x1] = True

        for cue_name, cue in cues(bgr, ksize).items():
            inside = cue[board_mask]
            outside = cue[~board_mask]
            mad = np.median(np.abs(outside - np.median(outside))) + 1e-6
            d = (np.median(inside) - np.median(outside)) / mad
            scores.setdefault(cue_name, []).append(d)

    print(f"\nSeparation (board vs background), {len(picks)} frames")
    print(f"{'cue':<18}{'median d':>11}{'min':>9}{'n>2':>7}")
    print("-" * 46)
    for cue_name, ds in sorted(scores.items(), key=lambda kv: -np.median(kv[1])):
        arr = np.array(ds)
        print(f"{cue_name:<18}{np.median(arr):>11.2f}{arr.min():>9.2f}"
              f"{int((arr > 2).sum()):>4}/{len(arr)}")


if __name__ == "__main__":
    main()
