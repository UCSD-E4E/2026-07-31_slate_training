"""Turn a board estimate into a Label Studio pre-annotation.

**The points come from the homography, not from the plane.** The plane is
3 DOF — it deliberately discards in-plane rotation and translation, which are
exactly the freedoms that decide where reference point 1 lands in the image.
`BoardEstimate` carries the homography for this reason: the plane and the
projected points are two different projections of the same fit, and only the
homography determines both.

Coordinate chain (see `contracts.py` for the space definitions):

    template render px  --H-->  rectified-photo px
                        --+panel_width-->  composite-canvas px
                        --/composite dims * 100-->  LS percentages

The panel width is recomputed here the way stage 9 actually builds the canvas,
including its integer truncation, so the declared `original_width` matches the
JPEG Label Studio is rendering byte-for-byte. A 1 px disagreement would shift
every seeded point.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from slate_training.contracts import Point

__all__ = [
    "composite_dimensions",
    "points_to_keypoint_results",
    "build_prediction",
]

KEYPOINT_LABEL = "Reference Point"
FROM_NAME = "reference_points"
TO_NAME = "image"


def composite_dimensions(
    pdf_width_px: int, pdf_height_px: int, photo_width: int, photo_height: int
) -> tuple[int, int, int]:
    """Return ``(panel_width, composite_width, composite_height)``.

    Mirrors `preprocess_slate_image.composite_slate_with_image` exactly:

        scale_y       = photo_height / pdf_height
        panel_width   = int(pdf_width * scale_y)      # truncated, not rounded
        canvas        = photo_width + panel_width  wide, photo_height tall

    The truncation is why the geometric panel width (4968.0) and the observed
    one (4967) differ by a pixel. Reproducing it here keeps predictions aligned
    with the canvas the labeler actually sees.
    """
    scale_y = float(photo_height) / float(pdf_height_px)
    panel_width = int(pdf_width_px * scale_y)
    return panel_width, photo_width + panel_width, photo_height


def points_to_keypoint_results(
    photo_points: Sequence[Point],
    panel_width: int,
    composite_width: int,
    composite_height: int,
    label: str = KEYPOINT_LABEL,
) -> List[Dict[str, Any]]:
    """Convert photo-space points into LS `keypointlabels` results.

    Label Studio stores keypoints as percentages of the rendered image, so the
    panel offset is added back (predictions live in composite space even
    though the model works in photo space) and the result is normalized.

    Points that fall outside the canvas are dropped rather than clamped — a
    clamped point silently lands on the canvas edge and reads to the labeler as
    a real detection.
    """
    results: List[Dict[str, Any]] = []
    for index, (x, y) in enumerate(photo_points):
        cx = float(x) + float(panel_width)
        cy = float(y)
        if not (0.0 <= cx < composite_width and 0.0 <= cy < composite_height):
            continue
        results.append(
            {
                "from_name": FROM_NAME,
                "to_name": TO_NAME,
                "type": "keypointlabels",
                "original_width": int(composite_width),
                "original_height": int(composite_height),
                "value": {
                    "x": 100.0 * cx / float(composite_width),
                    "y": 100.0 * cy / float(composite_height),
                    "width": 0.1,
                    "keypointlabels": [label],
                },
                "meta": {"template_index": index},
            }
        )
    return results


def build_prediction(
    photo_points: Sequence[Point],
    panel_width: int,
    composite_width: int,
    composite_height: int,
    score: float,
    model_version: str,
) -> Dict[str, Any] | None:
    """Build one LS `predictions` entry, or None if nothing should be seeded.

    Returns None when no point survives the canvas bounds. A prediction with a
    partial or empty point set is worse than no prediction: the labeler sees a
    confident-looking pre-annotation and may accept it, and the count mismatch
    would break the positional pairing stage 13 depends on
    (`contracts.template_correspondences`).

    Gating on confidence is the **caller's** job and must happen before this —
    see `docs/design.md`: below ECC ~0.80 the estimate is unreliable, and a bad
    pre-annotation is more harmful than none.
    """
    if not photo_points:
        return None
    results = points_to_keypoint_results(
        photo_points, panel_width, composite_width, composite_height
    )
    if len(results) != len(photo_points):
        return None

    return {
        "model_version": model_version,
        "score": float(score),
        "result": results,
    }
