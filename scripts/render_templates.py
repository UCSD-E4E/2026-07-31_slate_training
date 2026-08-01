"""Render each slate template and overlay its `reference_points`.

Establishes what the baseline is matching against: the printed pattern, where
the fiducials sit on it, and whether the board outline is part of the render
(which decides whether a quad detector has anything to lock onto).

`DiveSlate.reference_points` are in *rendered pixel* space at `DiveSlate.dpi`
-- that is the space stage 9 scales by `photo_height / pdf_height` when it
draws its numbered markers.

    uv run python scripts/render_templates.py
"""

import json
import os

import cv2
import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
NAS = os.path.expanduser("~/mnt/fishsense_data/REEF/data")
OUT = os.path.join(HERE, "..", "data", "templates")


def render(pdf_path, dpi):
    """Page 0 -> binarized BGR, matching preprocess_slate_image exactly."""
    with pymupdf.open(pdf_path) as doc:
        pix = doc.load_page(0).get_pixmap(dpi=dpi)
        raw = np.frombuffer(pix.samples, dtype=np.uint8)
        rgb = raw.reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binarized = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)


def main():
    slates = json.load(open(os.path.join(HERE, "..", "data", "slate_templates.json"),
                            encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)

    print(f"{'slate':<16}{'dpi':>5}{'render px':>14}{'n_pts':>7}"
          f"{'pts bbox (x,y)':>28}{'inside?':>9}")
    print("-" * 80)

    for s in sorted(slates, key=lambda r: r["id"]):
        img = render(os.path.join(NAS, s["path"]), s["dpi"])
        h, w = img.shape[:2]
        pts = np.array(s["reference_points"], dtype=float)

        inside = bool(
            (pts[:, 0] >= 0).all() and (pts[:, 0] < w).all()
            and (pts[:, 1] >= 0).all() and (pts[:, 1] < h).all()
        )
        print(f"{s['name']:<16}{s['dpi']:>5}{f'{w}x{h}':>14}{len(pts):>7}"
              f"{f'[{pts[:,0].min():.0f},{pts[:,0].max():.0f}] x [{pts[:,1].min():.0f},{pts[:,1].max():.0f}]':>28}"
              f"{'YES' if inside else 'NO':>9}")

        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (int(x), int(y)), 14, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (int(x) + 18, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        out = os.path.join(OUT, f"{s['name'].replace(' ', '_')}.png")
        scale = 900.0 / max(w, h)
        cv2.imwrite(out, cv2.resize(img, (int(w * scale), int(h * scale))))


if __name__ == "__main__":
    main()
