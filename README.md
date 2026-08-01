# slate-training

Model-assisted **dive-slate labeling** for the FishSense laser-calibration
pipeline — the last human bottleneck before a dive can be calibrated.

Given a rectified underwater frame and the known slate template, recover the
board's pose and emit a Label Studio pre-annotation, **gated** so that a
low-confidence frame produces *nothing* rather than a wrong pre-annotation.

## What this actually is (read this before integrating)

It is a **hybrid**, not the pure-classical or pure-learned thing the earlier
drafts of this file described:

| | |
| --- | --- |
| Geometry | **Classical** — template search → ECC → `solvePnP`. Bit-exact against stage 13. |
| Localization | **Small learned UNet** (1.08 M params) that says *where* the board is |
| Checkpoint | **Yes** — `board_unet.pt`, 4.4 MB, versioned like the laser detector |
| GPU | **No.** 202 ms/frame on CPU (4 threads). No `nodeAffinity`, no SM≥7.5. |

So: **package** like the laser detector (versioned checkpoint behind an
extra), **deploy** unlike it (ordinary CPU activity). The mask is optional —
`estimate_plane(..., board_mask=None)` is a supported path that costs ~13
points of coverage, so the activity still runs if the checkpoint is missing.

## Status and measured accuracy

Phase 2. Numbers are full-corpus (104 labels, 8 dives), gated at ECC ≥ 0.80,
evaluated with **out-of-fold** masks (each frame's mask from a model that never
saw its dive):

| | classical only | **+ learned mask** |
| --- | ---: | ---: |
| Frames seeded | 67% | **80%** |
| Median point error | 5.7 px | **5.9 px** |
| Median plane offset | 78 mm | **62 mm** |

Per family (seeded %): V-Slate 1 **89**, Tic-Tac-Toe **83**, V-Slate 3 **56**,
V-Slate 4 **56**. H-Slate is **unsupported** — it has zero labeled frames.

The gate is what makes this safe: ungated p90 point error is 1262 px, gated it
is 9.4 px. Failures are detectable, so the system declines rather than seeding
something wrong.

## Integration contract

One call. See [`predict.py`](src/slate_training/predict.py):

```python
predict_slate(bgr, template_gray, template_points, dpi, camera_matrix,
              pdf_width_px, pdf_height_px, photo_width, photo_height,
              slate_name, model_version, board_mask=None) -> SlatePrediction
```

`SlatePrediction` is either a ready-to-POST Label Studio `predictions` entry,
or `None` plus a `rejected_reason` (`no_board`, `low_confidence`,
`unsupported_slate_family`, `points_off_canvas`) so the worker can log *why*.

**Coordinates are handled here.** Predictions are emitted in composite-canvas
percentages, reproducing stage 9's integer truncation so `original_width`
matches the JPEG Label Studio renders. Verified end-to-end by replaying stage
12's own sync arithmetic: **5.6 px median round-trip** (`verify_end_to_end.py`).

This signature and the emitted JSON shape are **frozen**. Accuracy
improvements land as a new checkpoint and new threshold constants, not as API
changes.

## Two upstream bugs found

See [docs/upstream_bug_report.md](docs/upstream_bug_report.md). Both reported.

1. **All 104 stored labels are in composite coordinates** — stage 12's
   panel-width shift was never applied. Repairing it improves human-label
   reprojection from 1.06 → 0.64 px. Affects every existing `LaserExtrinsics`.
2. **Stage 13 mis-pairs points at ≥2 `skipped_points`** (latent today — no such
   label exists, but per-point visibility will create them).

## Layout

| Path | What |
| --- | --- |
| `src/slate_training/predict.py` | The integration entry point (gating + emission) |
| `src/slate_training/baseline.py` | Board localization and homography refinement |
| `src/slate_training/geometry.py` | The plane target; bit-exact vs stage 13 |
| `src/slate_training/contracts.py` | Coordinate/order contract, incl. the panel repair |
| `scripts/train_mask.py` | UNet training + leave-one-dive-out CV |
| `docs/design.md` | Every measurement, including what failed |

## Develop

```sh
uv sync --group dev
uv run pytest -q                      # 104 tests, no credentials or NAS needed
uv run python scripts/eval_baseline.py --oof-masks    # needs NAS + dump
```

Training (CUDA optional; on NixOS the driver libs need to be on the path):

```sh
LD_LIBRARY_PATH=/run/opengl-driver/lib \
  uv run --extra train python scripts/train_mask.py --cv --epochs 80
```
