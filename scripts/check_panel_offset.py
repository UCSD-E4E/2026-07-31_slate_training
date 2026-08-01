"""Derive the composite panel width per slate and confirm the stored labels
carry it.

Two independent derivations must agree:
  * geometric -- (pdf_width/pdf_height) * photo_height, what stage 12 computes
  * observed  -- LS `original_width` - photo_width, read off the task JSON

If they agree, the offset is exactly recoverable for every historical label
and the 104-row corpus can be repaired without re-labeling.
"""

import json
import os
import sys

import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
NAS = os.path.expanduser("~/mnt/fishsense_data/REEF/data")
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from slate_training.contracts import composite_panel_width  # noqa: E402

PHOTO_W, PHOTO_H = 4014, 3016


def main():
    slates = json.load(open(os.path.join(HERE, "..", "data", "slate_paths.json"),
                            encoding="utf-8"))
    rows = json.load(open(os.path.join(HERE, "..", "data", "slate_geometry_export.json"),
                          encoding="utf-8"))
    seen, uniq = set(), []
    for r in rows:
        if r["label_id"] in seen:
            continue
        seen.add(r["label_id"])
        uniq.append(r)

    print(f"photo = {PHOTO_W} x {PHOTO_H}\n")
    print(f"{'slate':<16}{'pdf pts':>14}{'aspect':>9}{'panel(geo)':>12}"
          f"{'composite_w':>13}{'x after shift':>20}")
    print("-" * 86)

    for name, path in sorted(slates.items()):
        labels = [r for r in uniq if r["slate_name"] == name]
        if not labels:
            continue
        with pymupdf.open(os.path.join(NAS, path)) as doc:
            rect = doc.load_page(0).rect
        aspect = rect.width / rect.height
        panel = composite_panel_width(aspect, PHOTO_H)

        x = np.concatenate([np.array(r["label_points"])[:, 0] for r in labels])
        lo, hi = x.min() - panel, x.max() - panel
        flag = "OK" if 0 <= lo and hi < PHOTO_W else "OUT OF BOUNDS"
        print(f"{name:<16}{rect.width:>7.0f}x{rect.height:<6.0f}{aspect:>9.4f}"
              f"{panel:>12.0f}{panel + PHOTO_W:>13.0f}"
              f"   [{lo:>7.0f},{hi:>7.0f}] {flag}")


if __name__ == "__main__":
    main()
