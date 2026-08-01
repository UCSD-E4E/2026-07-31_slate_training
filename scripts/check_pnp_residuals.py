"""Run stage 13's PnP on the *human* labels and report reprojection residuals.

This is the noise floor. Whatever a model produces has to be compared against
this, not against zero -- and an outlier residual on a human label is the only
practical detector for the click-order risk in docs/design.md §6.

Run with a venv that has cv2 + numpy:
    ../fishsense-lite/.venv/bin/python scripts/check_pnp_residuals.py
"""

import json
import os
import sys

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from slate_training.contracts import (  # noqa: E402
    composite_panel_width,
    drop_skipped,
    repair_panel_offset,
)

INCH_TO_M = 0.0254
PHOTO_W = 4014
HERE = os.path.dirname(os.path.abspath(__file__))
NAS = os.path.expanduser("~/mnt/fishsense_data/REEF/data")
EXPORT = os.path.join(HERE, "..", "data", "slate_geometry_export.json")
SLATE_PATHS = os.path.join(HERE, "..", "data", "slate_paths.json")

_PANEL_CACHE = {}


def panel_for(slate_name, photo_height=3016):
    """Composite panel width for a slate, from its PDF's intrinsic aspect."""
    if slate_name not in _PANEL_CACHE:
        paths = json.load(open(SLATE_PATHS, encoding="utf-8"))
        with pymupdf.open(os.path.join(NAS, paths[slate_name])) as doc:
            rect = doc.load_page(0).rect
        _PANEL_CACHE[slate_name] = composite_panel_width(
            rect.width / rect.height, photo_height
        )
    return _PANEL_CACHE[slate_name]


def normalize_skipped(raw):
    """`skipped_points` arrives as SQL NULL, JSON null, or a list."""
    return list(raw) if isinstance(raw, list) else []


def residual_for(row):
    """RMS reprojection error in pixels for one label, or None if PnP fails."""
    template = [tuple(p) for p in row["template_points"]]
    skipped = normalize_skipped(row["skipped"])
    # The stored corpus is in composite space; repair before any geometry.
    labeled = np.array(
        repair_panel_offset(
            [tuple(p) for p in row["label_points"]],
            panel_for(row["slate_name"]),
            PHOTO_W,
        ),
        dtype=np.float64,
    )

    source = drop_skipped(template, skipped)
    if len(source) != len(labeled):
        return None, "count_mismatch"

    body = np.zeros((len(source), 3), dtype=np.float32)
    body[:, :2] = (np.array(source) / float(row["dpi"])) * INCH_TO_M

    K = np.array(row["camera_matrix"], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(body, labeled, K, np.zeros((5,)))
    if not ok:
        return None, "pnp_failed"

    proj, _ = cv2.projectPoints(body, rvec, tvec, K, np.zeros((5,)))
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - labeled, axis=1)
    return float(np.sqrt((err**2).mean())), None


def main():
    rows = json.load(open(EXPORT, encoding="utf-8"))
    # The LEFT JOIN on laserlabel can duplicate a label row; dedupe by label_id.
    seen, uniq = set(), []
    for r in rows:
        if r["label_id"] in seen:
            continue
        seen.add(r["label_id"])
        uniq.append(r)

    by_slate = {}
    failures = []
    for r in uniq:
        rms, err = residual_for(r)
        if rms is None:
            failures.append((r["label_id"], err))
            continue
        by_slate.setdefault(r["slate_name"], []).append((rms, r))

    print(f"{len(uniq)} unique labels, {len(failures)} PnP failures\n")
    print(f"{'slate':<16}{'n':>4}{'median':>10}{'p90':>10}{'max':>10}")
    print("-" * 50)
    allv = []
    for slate in sorted(by_slate):
        v = np.array([x[0] for x in by_slate[slate]])
        allv.extend(v)
        print(
            f"{slate:<16}{len(v):>4}{np.median(v):>10.2f}"
            f"{np.percentile(v, 90):>10.2f}{v.max():>10.2f}"
        )
    allv = np.array(allv)
    print("-" * 50)
    print(
        f"{'ALL':<16}{len(allv):>4}{np.median(allv):>10.2f}"
        f"{np.percentile(allv, 90):>10.2f}{allv.max():>10.2f}"
    )

    print("\nWorst 12 labels (candidate mis-clicks / mis-ordered):")
    worst = sorted(
        ((rms, r) for lst in by_slate.values() for rms, r in lst),
        key=lambda t: -t[0],
    )[:12]
    for rms, r in worst:
        print(
            f"  label={r['label_id']:<6} dive={r['dive_id']:<5} "
            f"{r['slate_name']:<14} rms={rms:8.2f}px  "
            f"skipped={normalize_skipped(r['skipped'])} "
            f"upside_down={r['upside_down']}"
        )

    if failures:
        print(f"\nFailures: {failures[:20]}")


if __name__ == "__main__":
    main()
