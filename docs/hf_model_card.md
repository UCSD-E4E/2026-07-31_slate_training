---
license: mit
tags:
  - image-segmentation
  - underwater
  - photogrammetry
  - fishsense
library_name: pytorch
---

# fishsense-slate-detector

Board-localization mask for FishSense dive-slate labeling. Companion to
[`ucsde4e/fishsense-laser-detector`](https://huggingface.co/ucsde4e/fishsense-laser-detector).

Training code, evaluation, and the classical geometry this feeds:
[UCSD-E4E/2026-07-31_slate_training](https://github.com/UCSD-E4E/2026-07-31_slate_training)

## What it does — and what it deliberately does not

This model **only says where the calibration slate is** in a rectified
underwater frame. It does not find keypoints, estimate pose, or produce a
calibration. All of that stays classical (template search → ECC → `solvePnP`),
and is bit-exact against the production stage-13 calibration path.

That split is the point. The classical geometry was already accurate — gated
plane offsets 27–58 mm, 5.7 px points — while *localization* failed, because
the search has to scan the whole frame and reef texture competes with the
board. Adding search hypotheses made it worse, not better (a finer scale grid
dropped coverage 67% → 63%). A learned mask removes the competition instead of
trying to out-tune it.

## Usage

```python
from slate_training.mask import BoardMasker

masker = BoardMasker.from_pretrained()      # ucsde4e/fishsense-slate-detector
mask = masker.predict(bgr)                  # HxW float32 probabilities
```

**BGR input, not RGB** — training used `cv2.imread`, so channel order is baked
into the weights. Feeding RGB will not error; it will quietly produce a worse
mask.

**CPU only.** 202 ms/frame at 512×384 with 4 threads. 1.08 M params, 4.4 MB. No
GPU, no `nodeAffinity`, no SM≥7.5 requirement.

**Optional.** The downstream `predict_slate(..., board_mask=None)` path is
supported and costs ~13 points of coverage, so consumers can ship before
staging this checkpoint.

## Accuracy

Trained on 104 human-labeled frames across 8 dives. Masks were derived from
existing labels — a homography from the labeled correspondences warps the slate
template's page outline into each frame — so no new annotation was required.

Localization, **leave-one-dive-out** (never a frame-level split; frames within
a dive share board, camera, water and lighting):

| held-out dive | slate | centroid-hit |
| ---: | --- | ---: |
| 341 / 347 / 349 / 466 / 471 | V-Slate 1, V-Slate 4 | 1.00 |
| 383 | V-Slate 3 (sole dive) | 0.89 |
| 465 | V-Slate 1 | 0.80 |
| 279 | Tic-Tac-Toe 6 (sole dive) | 0.67 |
| **weighted** | | **0.95** |

End-to-end effect on the full pipeline, evaluated with **out-of-fold** masks
(each frame's mask from a model that never saw its dive):

| | classical only | + this checkpoint |
| --- | ---: | ---: |
| Frames auto-seeded | 67% | **80%** |
| Median point error | 5.7 px | 5.9 px |
| Median plane offset | 78 mm | **62 mm** |

## Limitations

* **The published weights are trained on all 104 frames with no holdout.**
  Their train-set metrics are not a generalization estimate; the honest numbers
  are the leave-one-dive-out figures above.
* **Weakest on board types with a single dive** — Tic-Tac-Toe (0.67) and
  V-Slate 3 (0.89) are effectively zero-shot under leave-one-dive-out. This
  improves as the corpus grows.
* **H-Slate is unsupported** — zero labeled frames exist. It shares the 8-point
  grid geometry and will probably behave like Tic-Tac-Toe, but that is an
  assumption, not a measurement.
* **Turbidity is unverified.** All 8 dives are clear-water Caribbean/Florida
  reef sites. The downstream confidence gate should fail safe, but that has not
  been tested in murky water.
* **Assisted labeling only, not auto-calibration.** Plane offsets are far from
  calibration-grade, and point accuracy does *not* imply plane accuracy (rank
  correlation +0.003 above the gate).
* The downstream confidence threshold (`ECC ≥ 0.80`) is only meaningful for the
  estimator configuration this checkpoint shipped with. Bump them together.

## Training

Small UNet, Dice+BCE (the board is a median **1.1%** of frame, so plain BCE is
minimised by predicting all-background), heavy augmentation including
per-channel colour jitter so the model learns the board rather than each site's
water cast.

```sh
uv run --extra train python scripts/train_mask.py --cv --epochs 80
```
