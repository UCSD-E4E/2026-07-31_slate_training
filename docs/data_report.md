# Data report — `DiveSlateLabel` corpus

Source: production dump `2026-07-31T03-00-01Z.dump`, restored locally.
All figures below are measured, not estimated.

## Headline

| Metric | Value |
| --- | --- |
| `diveslatelabel` rows (all) | 320 |
| **Completed, with points** | **104** |
| Superseded among completed | 0 |
| Dives represented | 8 |
| Slate templates defined | 11 |
| **Slate templates actually labeled** | **4 of 11** |
| Dives with a human `LaserExtrinsics` | 8 of 8 |
| Point-count contract violations | **0 of 104** |
| PnP failures on human labels | **0 of 104** |

The brief's estimates were accurate: ~104 labels, ~8 dives, ~70% V-Slate 1,
Tic-Tac-Toe nearly absent.

## Per-slate-type balance

| Slate | Labels | % | Dives | Template pts |
| --- | ---: | ---: | ---: | ---: |
| V-Slate 1 | 71 | 68.3% | 5 | 6 |
| V-Slate 4 | 18 | 17.3% | 1 | 6 |
| V-Slate 3 | 9 | 8.7% | 1 | 6 |
| Tic-Tac-Toe 6 | 6 | 5.8% | 1 | 8 |
| *H-Slate, Tic-Tac-Toe 1–5, V-Slate 2* | **0** | — | 0 | 8 / 6 |

**Seven of eleven templates have zero labels.** Three of the four labeled
types come from a single dive each — so "per-type model" is only meaningful
for V-Slate 1. Everything else is one-dive, and a per-type model there would
be fitting one dive's water, lighting, and camera.

All V-Slates have 6 reference points; H-Slate and all Tic-Tac-Toes have 8.
Every template is 300 dpi.

## Per-dive coverage (the real eval unit)

| Dive | Slate | Camera | Labels | Imgs w/ laser | Extrinsics |
| ---: | --- | ---: | ---: | ---: | ---: |
| 341 | V-Slate 1 | 1 | 33 | 33 | ✓ |
| 347 | V-Slate 4 | 2 | 18 | 17 | ✓ |
| 349 | V-Slate 1 | 6 | 12 | 12 | ✓ |
| 466 | V-Slate 1 | 6 | 11 | 11 | ✓ |
| 465 | V-Slate 1 | 4 | 10 | 10 | ✓ |
| 383 | V-Slate 3 | 3 | 9 | 9 | ✓ |
| 279 | Tic-Tac-Toe 6 | 5 | 6 | 6 | ✓ |
| 471 | V-Slate 1 | 3 | 5 | 5 | ✓ |

The oracle is **fully wired**: every dive has camera intrinsics, a human
`LaserExtrinsics`, and laser labels on all but one frame.

Since the split must be by dive (§4 of the design doc), the effective eval set
is **8 folds**, and only 5 of them are V-Slate 1. Dive 341 alone is a third of
the corpus. Any headline accuracy number will have wide error bars — report
per-dive, and treat a single dive's result as anecdote.

## Skipped points

| Skipped per label | Labels |
| ---: | ---: |
| 0 | 88 |
| 1 | 16 |
| ≥2 | **0** |

All 16 skip-bearing labels are V-Slate 1 (5 of 6 points visible).

**This makes the stage-13 `pop` bug latent, not active.** The sequential-pop
mis-pairing only manifests at ≥2 skipped points, and no such label exists, so
no current calibration is corrupted and the eval oracle is clean. The bug is
still real and should be fixed upstream before any auto-accept phase starts
producing multi-skip labels — a model with per-point visibility will generate
exactly that case.

Storage note: `skipped_points` is `[i]`, SQL `NULL`, or JSON `null` — three
encodings for "nothing skipped". Normalize on read.

## Reprojection noise floor (human labels)

Stage 13's own PnP over all 104 human labels, **after** the coordinate repair
below (`scripts/check_pnp_residuals.py`):

| Slate | n | median | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| V-Slate 4 | 18 | 0.52 px | 0.86 | 0.95 |
| V-Slate 1 | 71 | 0.64 px | 1.11 | 1.46 |
| Tic-Tac-Toe 6 | 6 | 0.66 px | 0.88 | 1.00 |
| V-Slate 3 | 9 | 1.42 px | 1.88 | 3.11 |
| **All** | **104** | **0.64 px** | **1.22** | **3.11** |

Two conclusions:

1. **Click order is reliable.** A mis-ordered or swapped click on a board this
   size produces residuals in the hundreds of pixels; the worst label in the
   corpus is 3.11 px. So `reference_points` really is in template order
   throughout, and the positional pairing stage 13 assumes is safe.
2. **The accept bar is ~1.5 px RMS.** Human median is 0.64 px, p90 1.22 px. A
   model at ≤1.5 px is within human labeling noise; >3.5 px is worse than any
   human label in the corpus and should never auto-accept.

Caveat: a planar target always admits an exact homography, so this residual
measures label *self-consistency*, not correctness — it cannot validate the
coordinate convention (see below, where it did not). It does reliably catch
permutation errors, which is what conclusion 1 rests on.

V-Slate 3 is the worst type but comes from a single dive (383), so board and
dive are confounded.

## Orientation

`upside_down` is `true` (35) or `NULL` (69) — **never `false`**. NULL means
the labeler left the box unchecked.

It is also almost perfectly confounded with dive: V-Slate 4 (dive 347) and
Tic-Tac-Toe 6 (dive 279) are 100% upside-down; V-Slate 3 is 8/9. An
orientation head trained on this would learn dive identity, not orientation.
Combined with the fact that nothing downstream consumes the field, this
confirms: **do not build an orientation head.**

## Images — available

The dump is metadata only, but the NAS at `~/mnt/fishsense_data/REEF/data`
covers everything:

| Asset | Resolved |
| --- | --- |
| Frames (`.ORF`) for labeled slates | **104 / 104** |
| Slate template PDFs | **11 / 11** |

Garage S3 (403 without credentials) is therefore not on the critical path.

## Coordinate defect: labels are in composite space

**All 104 labels are stored in composite coordinates** — stage 12's panel-width
subtraction was never applied to any row.

| Evidence | Value |
| --- | --- |
| Rectified photo | 4014 × 3016 |
| Stored `reference_points` x-range | **[5709, 7522]** — all beyond the photo |
| LS `original_width` | 8981 = 4967 panel + 4014 photo |
| Rows in photo bounds as stored | **0 / 104** |
| Rows in bounds after repair | **104 / 104** |

Panel width is per template family, derivable from the PDF aspect:

| Slate | PDF (pt) | Aspect | Panel @3016 |
| --- | --- | ---: | ---: |
| V-Slate 1 / 3 / 4 | 1008×612 | 1.6471 | 4968 px |
| Tic-Tac-Toe 6 | 792×612 | 1.2941 | 3903 px |

Repairing it improves every metric, which is independent confirmation:

| Metric | As stored | Repaired |
| --- | ---: | ---: |
| Median PnP RMS | 1.06 px | **0.64 px** |
| p90 | 2.72 px | **1.22 px** |
| Max | 7.45 px | **3.11 px** |

Visual proof in [`data/overlays/`](../data/overlays/): after repair the points
land exactly on the chevron tips (V-Slate) and grid intersections
(Tic-Tac-Toe). Fully recoverable **without re-labeling** —
`contracts.repair_panel_offset`.

Consequence: stage 13 currently fits poses from points ~4968 px outside the
image, and mixes spaces (laser labels *are* in photo pixels). The measured
impact on final extrinsics is modest — per-dive laser-ray collinearity shifts
by only a few mm — because the offset is largely absorbed as a board
translation. It is still systematically wrong and should be fixed.
