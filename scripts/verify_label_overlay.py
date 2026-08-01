"""Rectify real frames and overlay their human labels — the end-to-end
proof that the coordinate contract in `contracts.py` is right.

If `reference_points` really are rectified-photo pixels (i.e. the composite
panel offset was correctly removed by stage 12), the drawn dots land exactly on
the slate's physical fiducials. If the offset were still baked in, every dot
would sit far to the right of the board — a failure obvious at a glance.

Also reprojects the PnP-fitted template so predicted-vs-labeled can be
compared visually.

    uv run python scripts/verify_label_overlay.py --n 6
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import rawpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from slate_training.contracts import drop_skipped  # noqa: E402

INCH_TO_M = 0.0254
HERE = os.path.dirname(os.path.abspath(__file__))
NAS_ROOT = os.path.expanduser("~/mnt/fishsense_data/REEF/data")
EXPORT = os.path.join(HERE, "..", "data", "slate_geometry_export.json")
OUT_DIR = os.path.join(HERE, "..", "data", "overlays")


def rectify(orf_path, camera_matrix, distortion):
    """Decode an .ORF and undistort it — the space labels live in.

    Mirrors fishsense-core's RectifiedImage (RawImage -> cv2.undistort).
    rawpy's postprocess is used instead of the Rust raw path; that changes
    tone, not geometry, which is all this check cares about.
    """
    with rawpy.imread(orf_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return cv2.undistort(bgr, np.array(camera_matrix, float), np.array(distortion, float))


def annotate(img, row, panel_width):
    """Draw labeled points (green) and PnP-reprojected template (magenta).

    `panel_width` is subtracted from x because the stored labels are in
    *composite* coordinates -- stage 12's offset was never applied to the
    historical corpus. See docs/data_report.md.
    """
    template = [tuple(p) for p in row["template_points"]]
    skipped = row["skipped"] if isinstance(row["skipped"], list) else []
    labeled = np.array(row["label_points"], dtype=np.float64)
    labeled[:, 0] -= panel_width
    source = drop_skipped(template, skipped)

    body = np.zeros((len(source), 3), dtype=np.float32)
    body[:, :2] = (np.array(source) / float(row["dpi"])) * INCH_TO_M
    K = np.array(row["camera_matrix"], float)

    ok, rvec, tvec = cv2.solvePnP(body, labeled, K, np.zeros((5,)))
    proj = None
    if ok:
        proj, _ = cv2.projectPoints(body, rvec, tvec, K, np.zeros((5,)))
        proj = proj.reshape(-1, 2)

    for i, (x, y) in enumerate(labeled):
        cv2.circle(img, (int(x), int(y)), 18, (0, 255, 0), 4)
        cv2.putText(img, str(i + 1), (int(x) + 24, int(y) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5, cv2.LINE_AA)
    if proj is not None:
        for x, y in proj:
            cv2.drawMarker(img, (int(x), int(y)), (255, 0, 255),
                           cv2.MARKER_CROSS, 30, 4)

    if row.get("laser_x") is not None:
        cv2.drawMarker(img, (int(row["laser_x"]), int(row["laser_y"])),
                       (0, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 5)

    h, w = img.shape[:2]
    cv2.putText(img, f"label={row['label_id']} dive={row['dive_id']} {row['slate_name']}",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5, cv2.LINE_AA)
    return img, (w, h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()

    rows = json.load(open(EXPORT, encoding="utf-8"))
    seen, uniq = set(), []
    for r in rows:
        if r["label_id"] in seen:
            continue
        seen.add(r["label_id"])
        uniq.append(r)

    # One frame per slate type first, then fill out.
    by_slate = {}
    for r in uniq:
        by_slate.setdefault(r["slate_name"], []).append(r)
    picks = [v[0] for v in by_slate.values()]
    for v in by_slate.values():
        picks.extend(v[1:])
    picks = picks[: args.n]

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = json.load(open(os.path.join(HERE, "..", "data", "image_paths.json"),
                          encoding="utf-8"))
    paths, dist = meta["paths"], meta["dist"]
    slate_paths = json.load(open(os.path.join(HERE, "..", "data", "slate_paths.json"),
                                 encoding="utf-8"))

    import pymupdf

    for r in picks:
        rel = paths.get(str(r["image_id"]))
        if rel is None:
            print(f"no path for image {r['image_id']}")
            continue
        orf = os.path.join(NAS_ROOT, rel)
        img = rectify(orf, r["camera_matrix"], dist[str(r["image_id"])])
        with pymupdf.open(os.path.join(NAS_ROOT, slate_paths[r["slate_name"]])) as doc:
            rect = doc.load_page(0).rect
        panel = (rect.width / rect.height) * img.shape[0]
        img, dims = annotate(img, r, panel)
        out = os.path.join(OUT_DIR, f"label_{r['label_id']}_{r['slate_name'].replace(' ','_')}.jpg")
        cv2.imwrite(out, cv2.resize(img, (dims[0] // 3, dims[1] // 3)))
        print(f"wrote {out}  dims={dims}")


if __name__ == "__main__":
    main()
