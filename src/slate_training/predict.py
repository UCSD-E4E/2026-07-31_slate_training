"""Gated slate prediction — the unit a data-worker activity calls.

Ties the pieces together: estimate the board, decide whether the estimate is
trustworthy, and emit a Label Studio pre-annotation only if it is.

The gating is the important part. Measured over the real corpus
(`docs/design.md`), per-point accuracy against the human labels:

    ungated          p90 = 1262 px      <- unusable
    ECC >= 0.80      median 6.1 px, p90 7.6 px, 58% of frames kept

A bad pre-annotation is worse than none — the labeler sees a confident-looking
annotation and may accept it, and it lands in the corpus as ground truth. So
every rejection path here returns a *reason* rather than failing silently, and
the default posture is to decline.

Two gates, both from measurement:

1. **Slate family.** V-Slates and Tic-Tac-Toe are enabled; H-Slate is not.

   Tic-Tac-Toe was excluded at first because it landed ~81 px *while passing*
   the gate — confident and wrong. That turned out to be the whole-page
   template-patch bug, not the pattern: with the patch derived from the
   reference points, Tic-Tac-Toe became the **best** family measured (83%
   seeded, 5.1 px, 8.2 mm gated offset, 2.07 deg median plane angle, ECC
   0.955). Six frames, but the mechanism is understood and the margin is wide.

   **H-Slate stays excluded on zero evidence, not bad evidence** — it has no
   labeled frames at all. It shares the 8-point grid geometry, so it will
   probably behave like Tic-Tac-Toe, but "probably" is how the first exclusion
   went wrong in the other direction. Enable it when a labeled frame exists.
2. **Confidence.** ECC below `DEFAULT_MIN_CONFIDENCE`. Note this threshold is
   only meaningful *within* the current estimator configuration — the
   template-blur sweep showed ECC ranks configurations misleadingly even
   though it ranks frames well. Re-tune it if the estimator changes.

Deployment shape (this docstring previously said "no checkpoint", which is now
wrong — the learned mask was added afterwards):

* There **is** a versioned checkpoint: `board_unet.pt`, 1.08M params, 4.4 MB.
* It does **not** need a GPU: 202 ms/frame on CPU with 4 threads. No
  `nodeAffinity`, no SM>=7.5 requirement.
* The mask is **optional**. `board_mask=None` is a supported path that costs
  ~13 points of coverage (80% -> 67% seeded), so the activity still functions
  if the checkpoint is absent or fails to load.

So: package like the laser detector (versioned checkpoint behind an extra),
deploy unlike it (an ordinary CPU activity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

import numpy as np

from fishsense_core.slate import estimate_plane
from fishsense_core.plane import Plane
from slate_training.label_studio import build_prediction, composite_dimensions

__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "SUPPORTED_FAMILIES",
    "SlatePrediction",
    "slate_family",
    "predict_slate",
]

# ECC >= 0.80: 58% of frames kept at median 6.1 px / p90 7.6 px. Raising to
# 0.90 buys 3.3 px median but keeps only 10% — not worth it for assisted
# labeling, where coverage matters more than the last 3 px.
DEFAULT_MIN_CONFIDENCE = 0.80

SUPPORTED_FAMILIES = frozenset({"v-slate", "tic-tac-toe"})


@dataclass
class SlatePrediction:
    """Outcome of one frame: a seedable prediction, or a reason there isn't one."""

    prediction: Dict[str, Any] | None
    plane: Plane | None
    confidence: float
    rejected_reason: str | None


def slate_family(slate_name: str) -> str:
    """Coarse family key for a `DiveSlate.name` ('V-Slate 3' -> 'v-slate').

    Gating is per family rather than per board because the failure mode is a
    property of the pattern type — a chevron silhouette versus a grid — not of
    an individual taped board.
    """
    name = slate_name.strip().lower()
    if name.startswith("v-slate"):
        return "v-slate"
    if name.startswith("tic-tac-toe"):
        return "tic-tac-toe"
    if name.startswith("h-slate"):
        return "h-slate"
    return name


def predict_slate(
    bgr: np.ndarray,
    template_gray: np.ndarray,
    template_points: Sequence[Sequence[float]],
    dpi: float,
    camera_matrix: np.ndarray,
    pdf_width_px: int,
    pdf_height_px: int,
    photo_width: int,
    photo_height: int,
    slate_name: str,
    model_version: str,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    board_mask: np.ndarray | None = None,
    _estimator: Callable[..., Any] = estimate_plane,
) -> SlatePrediction:
    """Estimate the board and return a gated LS pre-annotation.

    `board_mask` is the learned board segmentation (any resolution, uint8 or
    float). Optional — omitting it falls back to purely classical localization
    at ~13 points lower coverage.

    **This signature is frozen.** Accuracy improvements ship as new checkpoints
    and new threshold constants, never as parameter changes, so a caller can
    integrate against it before the numbers stop moving.

    `_estimator` is injectable so the gating logic can be tested without
    OpenCV work; production callers should leave it alone.
    """
    if slate_family(slate_name) not in SUPPORTED_FAMILIES:
        return SlatePrediction(None, None, 0.0, "unsupported_slate_family")

    estimate = _estimator(
        bgr, template_gray, template_points, dpi, camera_matrix,
        board_mask=board_mask,
    )
    if estimate is None:
        return SlatePrediction(None, None, 0.0, "no_board")

    confidence = float(estimate.ecc_score)
    if confidence < min_confidence:
        return SlatePrediction(None, estimate.plane, confidence, "low_confidence")

    panel_width, composite_width, composite_height = composite_dimensions(
        pdf_width_px, pdf_height_px, photo_width, photo_height
    )
    prediction = build_prediction(
        [tuple(map(float, p)) for p in estimate.image_points],
        panel_width,
        composite_width,
        composite_height,
        confidence,
        model_version,
    )
    if prediction is None:
        # Points fell outside the canvas — see `build_prediction`, a partial
        # set would break stage 13's positional pairing.
        return SlatePrediction(None, estimate.plane, confidence, "points_off_canvas")

    return SlatePrediction(prediction, estimate.plane, confidence, None)
