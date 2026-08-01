"""End-to-end check: real frame -> predict_slate -> LS JSON -> back to pixels.

Closes the loop the way production will. Takes the emitted Label Studio
percentages and runs them back through exactly what stage 12 does on sync
(percent -> composite px -> subtract panel -> photo px), then compares against
the human label. If the round trip is lossy or the panel arithmetic is off by
a pixel, this catches it on real data rather than in a unit test's fixture.

    uv run python scripts/verify_end_to_end.py --stride 8
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
    composite_to_photo,
    repair_panel_offset,
)
from slate_training.predict import predict_slate, slate_family  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAS = os.path.expanduser("~/mnt/fishsense_data/REEF/data")
PHOTO_W, PHOTO_H = 4014, 3016
_TPL: dict = {}


def template(name, path, dpi):
    if name not in _TPL:
        with pymupdf.open(os.path.join(NAS, path)) as doc:
            page = doc.load_page(0)
            rect = page.rect
            pix = page.get_pixmap(dpi=dpi)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, binar = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        _TPL[name] = (binar, pix.width, pix.height,
                      composite_panel_width(rect.width / rect.height, PHOTO_H))
    return _TPL[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=8)
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
    uniq = uniq[:: args.stride]

    seeded = declined = 0
    reasons: dict = {}
    errors = []

    for r in uniq:
        name = r["slate_name"]
        tpl_meta = templates[name]
        tpl_gray, pdf_w, pdf_h, panel_geo = template(
            name, slate_paths[name], tpl_meta["dpi"]
        )
        K = np.array(r["camera_matrix"], float)

        with rawpy.imread(os.path.join(NAS, meta["paths"][str(r["image_id"])])) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        bgr = cv2.undistort(
            np.ascontiguousarray(rgb[:, :, ::-1]), K,
            np.array(meta["dist"][str(r["image_id"])], float),
        )

        out = predict_slate(
            bgr=bgr, template_gray=tpl_gray,
            template_points=tpl_meta["reference_points"], dpi=tpl_meta["dpi"],
            camera_matrix=K, pdf_width_px=pdf_w, pdf_height_px=pdf_h,
            photo_width=PHOTO_W, photo_height=PHOTO_H,
            slate_name=name, model_version="classical-v1",
        )
        if out.prediction is None:
            declined += 1
            reasons[out.rejected_reason] = reasons.get(out.rejected_reason, 0) + 1
            continue
        seeded += 1

        # Replay stage 12's sync arithmetic on our own emitted JSON.
        results = out.prediction["result"]
        width = results[0]["original_width"]
        height = results[0]["original_height"]
        composite = [(v["value"]["x"] / 100.0 * width,
                      v["value"]["y"] / 100.0 * height) for v in results]
        recovered = composite_to_photo(composite, width - PHOTO_W)

        truth = np.array(
            repair_panel_offset([tuple(p) for p in r["label_points"]],
                                panel_geo, PHOTO_W), dtype=float)
        skipped = r["skipped"] if isinstance(r["skipped"], list) else []
        keep = [i for i in range(len(recovered)) if i not in set(skipped)]
        if len(keep) != len(truth):
            continue
        d = np.linalg.norm(np.array(recovered)[keep] - truth, axis=1)
        errors.append(np.median(d))
        print(f"  {name:<14} label={r['label_id']:<5} conf={out.confidence:.3f} "
              f"round-trip median {np.median(d):6.1f}px")

    total = seeded + declined
    print(f"\nseeded {seeded}/{total} ({100*seeded/max(1,total):.0f}%), "
          f"declined {declined}")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"   declined: {reason:<28} {count}")
    if errors:
        e = np.array(errors)
        print(f"\nround-trip error over seeded frames: "
              f"median {np.median(e):.1f}px  p90 {np.percentile(e, 90):.1f}px  "
              f"max {e.max():.1f}px")


if __name__ == "__main__":
    main()
