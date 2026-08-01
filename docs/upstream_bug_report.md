# Upstream bug report — dive-slate label coordinates & PnP subsetting

Two defects found in `fishsense-lite` while building the slate-detector
training set. Both were found by reading production data, not by reading code;
evidence is included so you can reproduce independently.

Source: production dump `2026-07-31T03-00-01Z.dump`, restored locally.
Frames/templates from the NAS at `REEF/data`.

---

## Bug 1 — the composite panel offset is not applied to stored slate labels

**Severity: affects every existing `LaserExtrinsics`.**

### What's wrong

The slate labeling task in Label Studio is a composite canvas: the slate
template PDF rendered on the left, the rectified photo on the right
(`preprocess_slate_image.composite_slate_with_image`). Labelers click on the
photo half, so raw LS coordinates are offset horizontally by the rendered
panel width.

`sync_dive_slate_labels_for_label_studio_project_activity` has correct logic to
remove that offset (`compute_pdf_panel_width_in_composite` + `_shift_x`).
**It did not run for any label currently in the database.**

### Evidence

All 104 completed `diveslatelabel` rows (of 320 total), across all 8 dives:

| Fact | Value |
| --- | --- |
| Rectified photo dimensions | 4014 × 3016 |
| Stored `reference_points` x-range | **[5709, 7522]** |
| Rows with any point inside the photo | **0 / 104** |
| LS `original_width` in `label_studio_json` | 8981 = 4967 panel + 4014 photo |

Every stored point lies beyond the right edge of the image it supposedly
annotates.

Subtracting the panel width fixes all of them, and improves label
self-consistency across the board — independent confirmation the shift is the
right one:

| Metric (stage 13's own `solvePnP`, all 104 labels) | As stored | Shifted |
| --- | ---: | ---: |
| Median reprojection RMS | 1.06 px | **0.64 px** |
| p90 | 2.72 px | **1.22 px** |
| Max | 7.45 px | **3.11 px** |
| Points inside frame | 0 / 104 | **104 / 104** |

Visually, after the shift the points land exactly on the physical fiducials
(V-Slate chevron tips, Tic-Tac-Toe grid intersections). Before, they land
nowhere in the frame.

### The offset is per template family

`panel_width = pdf_width × (photo_height / pdf_height)`:

| Slate | PDF (pt) | Panel @ 3016 px tall |
| --- | --- | ---: |
| V-Slate 1 / 3 / 4 | 1008 × 612 | 4967.5 px |
| Tic-Tac-Toe 6 | 792 × 612 | 3903.1 px |

Geometric and observed agree to ~1 px (the compositor's `int()` truncation on
`new_pdf_width`). **A single global constant would corrupt Tic-Tac-Toe dives** —
the repair must be keyed per template.

### Impact

1. `perform_laser_calibration_activity` feeds these points to `cv2.solvePnP`,
   so it is fitting board poses from points ~4968 px outside the image. It
   converges with a low residual — a planar target always admits an exact
   homography — but on a geometrically wrong pose. Note this means a low
   reprojection error is *not* evidence the pipeline is correct.
2. Stage 13 currently mixes coordinate spaces: slate `reference_points` are in
   composite pixels while `LaserLabel.x/y` are correctly in photo pixels
   (verified by overlay).
3. Measured impact on final extrinsics is modest — repairing the offset changes
   per-dive laser-ray collinearity by a few mm, mostly improving — because the
   shift is largely absorbed as a board translation. It is still systematic and
   wrong.

### Two things to determine

- **Root cause.** Either these rows were written by the stage-12 *notebook*
  before the offset logic was ported into the activity, or they hit the
  activity's silent fallback:
  ```python
  except (ClientError, BotoCoreError) as e:
      activity.logger.warning("Could not fetch slate PDF ...; skipping panel-width offset")
      return None      # -> panel_width stays 0.0
  ```
  If it's the latter, the fallback is dangerous: a transient Garage failure
  silently produces labels in the wrong space with no downstream signal. It
  should fail the label rather than persist bad geometry.
- **Whether to recompute the 8 existing `LaserExtrinsics`** after repairing.
  They were all fitted from composite-space points.

### Suggested fix

1. Backfill: repair the 104 rows in place. `x -= panel_width` per template,
   idempotent (skip rows already in bounds), and reject any shift landing
   outside `[0, photo_width)` rather than writing a negative coordinate.
2. Make the PDF-fetch fallback fail loudly instead of silently zeroing the
   offset.
3. Add a validation tripwire: a slate label whose points fall outside the
   frame is always a bug.

No re-labeling is needed — the labelers clicked correctly, the offset removal
just never happened.

---

## Bug 2 — `skipped_points` subsetting mis-pairs at ≥2 skipped points

**Severity: currently latent. Will activate with model-assisted labeling.**

### What's wrong

`perform_laser_calibration_activity._laser_point_in_camera_space` (and
`scripts/stage13_perform_laser_calibration.ipynb`, which it was ported from):

```python
source_points = list(slate.reference_points or [])
for idx in label.skipped_points or []:
    source_points.pop(idx)
```

`list.pop` mutates as it goes, so indices after the first are interpreted
against the already-shortened list. With `skipped_points = [2, 5]` on an
8-point template:

| | result |
| --- | --- |
| current `pop` | `0 1 3 4 `**`5`**` 7` |
| correct | `0 1 3 4 `**`6`**` 7` |

Correct for 0 or 1 skipped point; wrong for ≥2. Each affected label feeds
`solvePnP` a wrong 3D↔2D correspondence.

### Why it's latent today

Measured over the 104 completed labels:

| Skipped per label | Count |
| ---: | ---: |
| 0 | 88 |
| 1 | 16 |
| **≥2** | **0** |

So no current calibration is affected.

### Why it matters anyway

The slate detector will emit per-point visibility, which is exactly what
produces multi-skip labels. The bug goes live the moment model-assisted
labeling starts, and it fails silently — biased extrinsics, no error.

### Suggested fix

Resolve all indices against the original list:

```python
drop = set(label.skipped_points or [])
source_points = [p for i, p in enumerate(slate.reference_points) if i not in drop]
```

Plus validation: reject duplicate or out-of-range indices, and assert
`len(reference_points) == len(template) - len(skipped)` — a mismatch silently
mis-pairs every correspondence.

---

## Also worth knowing

- `skipped_points` has three encodings for "nothing skipped": `[]`, SQL `NULL`,
  and JSON `null` (79 / 9 / 16 split). Normalize on read.
- `upside_down` is only ever `true` or `NULL` — never `false`. It is also read
  by nothing downstream; the only use in the stage-13 notebook is commented
  out. Orientation is already encoded in the point *ordering*.
- Ordering itself is sound: worst reprojection residual across the corpus is
  3.11 px, where a mis-ordered click would produce hundreds.

## Reference implementation

Order-safe subsetting, idempotent panel repair, and the contract checks are
implemented and tested in the slate-training repo:
`src/slate_training/contracts.py` (`repair_panel_offset`, `drop_skipped`,
`template_correspondences`), with reproductions in
`scripts/check_pnp_residuals.py`, `scripts/check_panel_offset.py`, and
`scripts/verify_label_overlay.py`.
