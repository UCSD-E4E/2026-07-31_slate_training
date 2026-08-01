# Slate detector — Phase 0 findings and approach

Status: **Phase 0. Metadata measured; images still blocked.** Contract verified
against `fishsense-lite` @ 2026-07-31; corpus measured from the production dump
`2026-07-31T03-00-01Z.dump`. Numbers live in
[data_report.md](data_report.md).

---

## 1. What the label actually is

Reading the live pipeline changed three things in the original brief. All three
are load-bearing for the model design.

### 1.1 The labeler works on a composite canvas

Stage 9 (`preprocess_slate_image.py`) does not upload the rectified photo. It
builds a **composite**: the slate template PDF rendered and binarized on the
left, the rectified photo concatenated on the right, with numbered red markers
drawn on the template panel.

```
┌─────────────┬──────────────────────────┐
│  template   │   rectified photo        │
│  render     │                          │
│  (numbered  │   ← labeler clicks here  │
│   markers)  │                          │
└─────────────┴──────────────────────────┘
 panel_width = (pdf_w/pdf_h) * photo_height
```

The template panel is the labeler's *legend* — it tells them which physical
fiducial is point 1, 2, 3…, which is how an ordered label is possible at all.

Stage 12 (`sync_dive_slate_labels_...`) is *supposed* to subtract
`panel_width` before persisting — see §1.5, it does not in practice.

### 1.2 `slate_rectangle` is a 2-point bbox, not a quad

The brief assumed a 4-corner quad. The LS control is `RectangleLabels`, and the
sync stores exactly two points — `[(x0,y0), (x1,y1)]`, top-left and
bottom-right of an axis-aligned box. Rotation is not read.

**Consequence: there is no labeled quad to fit a homography to.** The brief's
recommended two-stage design ("segment board → 4-corner quad → homography →
project template points") has no supervision for its first stage and no
labeled target to evaluate it against. The pose prior must instead come from
the reference points themselves — which is exactly what stage 13 already does
via `solvePnP`. `slate_rectangle` is usable only as a coarse ROI/attention
crop, and it is `None` on some rows.

### 1.3 `upside_down` is captured but never consumed

No downstream math reads it. In the stage-13 notebook the only use is
commented out:

```python
# if any(slate.upside_down for slate in dive_slate_labels):
#     continue
```

It is a labeler-facing flag (it tells them which end of the board is "point
1"), so orientation is already folded into the *ordering* of
`reference_points`. An `upside_down` head is therefore a pre-annotation UX
nicety, not a calibration input — and orientation correctness shows up
automatically in the calibration metric via point ordering. Deprioritize it.

### 1.5 Production bug: the panel offset was never applied

**All 104 completed labels are stored in composite coordinates.** Measured, not
inferred:

- the rectified photo is 4014 × 3016;
- stored `reference_points` have x in **[5709, 7522]** — every one beyond the
  right edge of the photo;
- the LS task JSON records `original_width = 8981`, i.e. 4967 px of panel plus
  4014 px of photo.

Subtracting the panel width puts every point back inside the frame, and
[the overlays](../data/overlays/) show them landing exactly on the physical
fiducials. Three independent confirmations:

| Check | Raw (as stored) | Repaired |
| --- | ---: | ---: |
| Median PnP reprojection RMS | 1.06 px | **0.64 px** |
| p90 / max | 2.72 / 7.45 px | **1.22 / 3.11 px** |
| Points inside frame | 0 / 104 | **104 / 104** |

The offset is **per template family**, derivable from the PDF's intrinsic
aspect: 4968 px for the V-Slates (1008×612 pt), 3903 px for Tic-Tac-Toe 6
(792×612 pt). Geometric and observed agree to 1 px (the compositor's `int()`
truncation).

Why it happened: the offset logic lives in the ported *activity*, but this
corpus was written by the stage-12 *notebook* that predates it — and the
activity also has a silent fallback (`except ClientError: → skip the offset`)
that would reproduce the same result if the slate PDF fetch fails.

Consequences:

1. **Stage 13 is fitting board poses from points ~4968 px outside the image.**
   Because a homography always exists for a planar target, `solvePnP` still
   converges with a low residual — it just converges on a geometrically wrong
   pose. Every existing `LaserExtrinsics` is affected.
2. It also means slate points and laser points are currently in **different
   coordinate spaces** (laser labels *are* in photo pixels — verified visually),
   so stage 13 mixes conventions.
3. Empirically the impact on the final extrinsics is modest — repairing the
   offset changes per-dive laser-ray collinearity by only a few mm, mostly for
   the better — because the shift is largely absorbed as a board translation.
   It should still be fixed: it is wrong, and the error is systematic.

`contracts.repair_panel_offset` implements the idempotent repair, and the
extractor applies it. This is recoverable **without re-labeling**.

### 1.4 The ordering contract

`reference_points` is a **dense list positionally aligned** with
`DiveSlate.reference_points` after `skipped_points` are removed. Stage 13
pairs them by position only. A count mismatch mis-pairs 3D↔2D silently, so
`contracts.template_correspondences` makes it a hard error and the extractor
quarantines those rows.

---

## 2. Upstream bug found: multi-skip point mis-pairing

`perform_laser_calibration_activity._laser_point_in_camera_space` (and the
stage-13 notebook it was ported from) subsets the template with a sequential
pop:

```python
source_points = list(slate.reference_points or [])
for idx in label.skipped_points or []:
    source_points.pop(idx)
```

Popping index 2 slides every later element down one, so a subsequent pop of 5
removes what was originally index **6**. Verified:

| template (8 pts), skipped `[2, 5]` | result |
| --- | --- |
| stage 13 `pop` | `0 1 3 4 **5** 7` |
| order-safe | `0 1 3 4 **6** 7` |

Correct for 0 or 1 skipped point; **wrong for ≥2**. Every affected label feeds
`solvePnP` one or more wrong 3D↔2D correspondences, biasing that frame's pose
and therefore the dive's `LaserExtrinsics`.

**Measured: the bug is latent, not active.** Of 104 completed labels, 88 skip
nothing and 16 skip exactly one point; **none skip two or more**. So no current
calibration is corrupted and the eval oracle is clean — an earlier concern that
the data has now retired.

It should still be fixed upstream before Phase 4. A model with per-point
visibility is precisely what will start emitting multi-skip labels, at which
point the bug goes live and silently biases the calibrations the autonomy phase
is meant to trust.

`contracts.drop_skipped` implements the order-safe version and its test pins
the difference.

---

## 3. Approach

### The target is the board *plane* — 3 DOF

The specific `reference_points` are an **arbitrary parameterization**, and so,
it turns out, is most of the pose. Stage 13's kernel uses exactly two things:

```python
slate_normal = rotation[:, 2]
scale = (slate_normal.T @ camera_space_points[0, :]) / (slate_normal.T @ ray)
```

— the board's normal and its offset along that normal. Measured on real
labels (`src/slate_training/geometry.py`, tests in `tests/test_geometry.py`):

| Perturbation of the fitted pose | Change in recovered laser point |
| --- | ---: |
| In-plane rotation 17° / 90° / **180°** | **0.000000 mm** |
| In-plane slide 5 cm / 50 cm | **0.000000 mm** |
| Motion 1 cm *along the normal* | 10.206 mm |

**So the deliverable is a plane `n · X = d` in camera space — 3 DOF.** Not
points, and not a 6-DOF pose. Three consequences:

1. **`upside_down` provably cannot affect the calibration.** A 180° in-plane
   flip yields an identical plane. That is the mathematical reason nothing
   downstream reads the field — and it retires the orientation question
   entirely, rather than merely deprioritizing it.
2. **In-plane correspondence errors are harmless.** Any mis-assignment that
   amounts to an in-plane symmetry leaves the plane untouched, so the estimate
   is far more robust than a point-matching framing suggests.
3. **Scale still matters.** `d` is metric, so the template's physical geometry
   (`reference_points / dpi → inches → metres`) stays essential. A homography
   alone recovers the plane only up to scale.

`geometry.py` reproduces stage 13's laser points **bit-exactly** (max deviation
0.0000 nm over all 103 laser-bearing labels), so it is a drop-in equivalent
formulation, not an approximation.

Since the board is planar and its printed geometry is fully known (template
pixels ÷ `dpi` → inches → metres), **one homography `H: template → image`
determines everything**:

```
rectified photo
   → estimate H (template plane -> image)
   → sample template points through H  (canonical reference_points, or denser)
   → solvePnP against known metric geometry -> pose
   → reprojection residual + inlier count -> confidence
```

Consequences worth being explicit about:

1. **We emit the canonical `reference_points` by projecting them through `H`.**
   The label format is satisfied exactly and self-consistently, and ordering is
   correct by construction — there is no ordering to get wrong, and no
   per-point matching problem to solve.
2. **Occlusion mostly stops mattering.** A fiducial hidden behind a hand is
   still recoverable by projection, because `H` is fit from whatever else is
   visible. `skipped_points` becomes an honesty signal for the human reviewer
   rather than a constraint on the geometry. (It also means we may emit *zero*
   skips in the common case — which is why upstream Bug 2 is low-urgency for
   us, though still worth fixing.)
3. **Far more of the image becomes usable evidence.** Rather than regressing 6
   or 8 discrete points, `H` can be fit from the entire printed pattern —
   every edge and stroke of the chevron or grid. That is a much better
   conditioned estimate from scarce data than 6 independent heatmap peaks.

This also dissolves the awkward 6-vs-8-point head design: there is one output
(a homography, or a segmentation the homography is fit to), independent of
template.

### How to estimate `H`

In increasing order of cost — stop as soon as the calibration metric is met:

1. **Contour/shape alignment.** Segment the black pattern, align to the
   binarized template render. For the V-Slates the chevron is one connected
   polygon, so a direct 6-vertex correspondence already over-determines `H`.
2. **Dense direct alignment.** `cv2.findTransformECC` with a homography motion
   model, initialized from (1). Uses every pixel of the pattern and refines to
   subpixel without needing discrete corners at all.
3. **Learned segmentation → (1)+(2).** If classical segmentation fails on
   glare/turbidity, replace only the segmentation step with a UNet — reusing
   the laser detector's stack — and keep the geometry classical. This is a far
   smaller learning problem than keypoint regression, and its training target
   (a board mask) is cheap to synthesize.

Conditioning on slate type is free — `dive_slate_id` is known per dive, and the
template render is available at predict time.

**The measured corpus rules out per-type models.** Only 4 of 11 templates have
any labels at all, and three of those four come from a single dive each (18, 9,
and 6 labels). A per-type model there would fit one dive's water, lighting, and
camera rather than the board. Only V-Slate 1 (71 labels / 5 dives) could
support one, and even it is thin.

If a learned component is needed at all, it should therefore be **one shared
board-segmentation model** — template-agnostic, since "white board with black
marks" is common to all 11 — with the template-specific geometry supplied
classically at fit time. That is the smallest learning problem consistent with
the data we have, and it sidesteps the 6-vs-8-point head question entirely.

### Classical baseline first — still the right call, and better-posed than expected

Rendering all 11 templates (`scripts/render_templates.py`, output in
`data/templates/`) shows the reference points are not arbitrary marks — they
are **corner features of the printed pattern itself**:

| Family | Pattern | Reference points are… |
| --- | --- | --- |
| V-Slate 1–4 | one solid black chevron | the **6 polygon vertices** of the chevron |
| H-Slate, Tic-Tac-Toe 1–6 | a hand-drawn `#` grid | the **8 inner intersections** where strokes cross |

That is a much stronger starting position than generic template/feature
matching. The baseline becomes:

```
rectified photo
   → segment the high-contrast black pattern on the white board
   → extract corners (polygon approx for V-Slates; stroke-crossing
     detection for the grids)
   → match to template order via the homography that best fits
   → refine subpixel (cornerSubPix) → solvePnP
```

Zero training cost, works on all 11 templates immediately, and produces the
same PnP-ready output as the model. This is the honest bar the learned model
must beat, and it may simply suffice.

#### Measured result: the geometry works, the localization does not

Four attempts, each ending in a measurement rather than a guess. The geometry
half is sound and unit-tested (quad ordering, homography seeding, ECC at
reduced resolution and its lift back to full scale — 60 tests). Localization
is where it fails, and the reason is now precise.

**Step 1 — brightness is worthless.** Board median grey **101** against a frame
median of **102**; saturation **165 vs 166**. The white slate is
photometrically identical to the water column. Otsu selects reef or open
water; plane error ~65°, ~2 m.

**Step 2 — cue probe** (`scripts/probe_cues.py`, d′ against background MAD,
8 frames spanning all four labeled templates):

| cue | median d′ | min | frames > 2 |
| --- | ---: | ---: | --- |
| **local_std** | **12.00** | 9.02 | **8/8** |
| local_range | 5.39 | 3.18 | 8/8 |
| grad_density | 3.33 | 1.44 | 6/8 |
| grey-world lowS | 0.02 | −0.11 | 0/8 |
| grey *(control)* | −0.02 | −2.21 | 2/8 |
| saturation *(control)* | −0.06 | −2.00 | 0/8 |
| grey-world V | −0.31 | −0.81 | 2/8 |

Two firm conclusions: **local contrast is the cue** (d′ = 12 on every frame),
and **underwater colour correction is a dead end** (d′ ≈ 0 — the attenuated red
channel does not hide a recoverable white board).

**Step 3 — the cue is necessary but not sufficient.** Rebuilt on `local_std`,
the detector still returned ~63°. Direct measurement of the mask explains it:

> The mask is correctly **on at the board centre in 7 of 8 frames** — but it
> also covers **18–26% of the whole frame**, because reef texture is just as
> locally contrasty as a printed pattern. The board's blob merges with the reef
> into one giant connected component, leaving the nearest candidate 250–550 px
> from the true centre when the board is only ~280 px across.

**Step 4 — template-scored candidate selection doesn't rescue it.** Scoring
every (region × rotation) by correlation and refining the best three raised the
ECC score (0.18 → 0.51) while leaving plane error flat at ~63°: proof the
template was aligning confidently to *reef*, because the board was never a
separate candidate to choose.

**Step 5 — multi-scale template search fixes localization.** Replacing
connected components with an exhaustive sliding search over scale × in-plane
rotation (`baseline.template_search`):

| stage | median angle | median offset | median ECC |
| --- | ---: | ---: | ---: |
| blob candidates | 63.9° | 2159 mm | 0.51 |
| **+ template search** | **20.1°** | **65 mm** | **0.89** |

A 33× improvement in offset, and an ECC of 0.89 means the template is genuinely
locked onto the pattern rather than confidently aligned to reef. A sliding
search has no segmentation step to fail — it scores every location
independently, so the board cannot be absorbed into a larger blob. Out-of-plane
foreshortening is deliberately *not* searched: `matchTemplate` tolerates
moderate perspective and the ECC stage recovers the full homography from a
fronto-parallel seed.

The enabling detail is that the search runs on a **locally normalized** image.
Matching raw grey finds nothing, for the §Step-1 reason.

#### Negative results worth not repeating

| tried | outcome |
| --- | --- |
| Grey-world / underwater colour correction | d′ ≈ 0. No recoverable white board. |
| Coarse-to-fine ECC (2×, 4× passes) | Bit-identical; the fine pass never beat the coarse score. |
| Normalizing the image before **ECC** | **Worse**: 20.1° → 25.5°, 65 mm → 142 mm. |
| Full-resolution crop refinement | **Worse**: 127 mm → 216 mm corpus-wide. |
| Blurring the template to match the image | **Worse, monotonically** (see below). |

Two of these are instructive beyond "don't do it".

**Finding the board and refining onto it want different representations.**
Finding needs local normalization to defeat the global brightness problem;
refining needs raw local structure, because the normalization window is
comparable to the tape strokes' width and suppresses exactly the edges ECC
locks onto.

**ECC score is not a proxy for plane accuracy.** The template-blur sweep makes
this unambiguous:

| blur σ | ALL offset | V-Slate 1 offset | ECC |
| ---: | ---: | ---: | ---: |
| **0** | **122.1 mm** | **84.9 mm** | 0.889 |
| 1 | 126.4 mm | 92.8 mm | 0.894 |
| 2 | 133.1 mm | 107.5 mm | **0.903** |
| 4 | 1761 mm | 2194 mm | 0.446 |

Accuracy degrades monotonically while the correlation score *rises* — a softer
template correlates more easily with everything, water and coral included. Any
confidence gate built on ECC must therefore be calibrated **within a fixed
configuration**; comparing scores across configurations is actively
misleading, and "the score went up" is not evidence a change helped.

#### Final classical result (104 frames, blur = 0)

| slate | n | med angle | med offset | p90 offset | med ECC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tic-Tac-Toe 6 | 6 | **9.28°** | 605 mm | 2179 | 0.731 |
| V-Slate 1 | 71 | 26.86° | **90 mm** | 2657 | 0.868 |
| V-Slate 3 | 9 | 63.52° | 1607 mm | 2546 | 0.686 |
| V-Slate 4 | 18 | 58.38° | 658 mm | 3416 | 0.658 |
| **all** | 104 | 34.15° | 127 mm | 2713 | — |

Tic-Tac-Toe's eight scattered tape corners give by far the best *normal*
(9.28° vs 27–63°), as predicted — well-distributed points constrain a plane
far better than a 6-vertex chevron whose silhouette barely changes under
moderate out-of-plane rotation. Only 6 frames, so directional.

#### The confidence gate does work — within a fixed configuration

Rank correlation over the 104 frames at the chosen configuration:

| pair | correlation |
| --- | ---: |
| ECC vs offset error | **−0.731** |
| ECC vs angle error | −0.415 |
| reprojection RMS vs offset error | +0.393 |

And the two populations separate cleanly: frames with offset < 200 mm have ECC
median 0.889 / p10 **0.847**, while the rest have median 0.646 / p90 **0.751**.
There is a real gap to put a threshold in.

| gate | frames kept | med offset | **p90 offset** | med angle |
| --- | ---: | ---: | ---: | ---: |
| none | 104 (100%) | 127 mm | **2713 mm** | 34.2° |
| ECC ≥ 0.80 | 55 (53%) | **61 mm** | **122 mm** | 26.7° |
| ECC ≥ 0.85 | 49 (47%) | 62 mm | 122 mm | 24.5° |
| ECC ≥ 0.90 | 10 (10%) | 65 mm | 123 mm | **8.5°** |

**The catastrophic tail is entirely gate-detectable.** p90 collapses from
2713 mm to 122 mm at ECC ≥ 0.80, at the cost of declining ~47% of frames. That
is exactly the failure mode assisted labeling can absorb: the estimator either
proposes a good board or says nothing, and never silently proposes a bad one.

Note this does not contradict the blur finding — ECC ranks *frames* reliably
within one configuration while ranking *configurations* misleadingly. Both
statements are needed; only the first is what a gate uses.

#### The 180° search gap — the largest single fix

The template search originally swept in-plane rotation over 0–180°, on the
implicit assumption that half a turn suffices. It does not: the tape patterns
are **haphazard by design and not 180°-symmetric** — that asymmetry is exactly
what makes board orientation legible to a labeler. A chevron held at 225°
therefore has *no match anywhere in the search space*, and the search settles
on background instead.

Found by looking at failures rather than measuring proxies. Rank correlation
said close boards fail (distance +0.326), and the obvious explanation —
close boards are more oblique — was **wrong** (span↔obliquity correlation
−0.115, and obliquity barely predicts outcome). Rendering the worst declines
showed boards that were large, close, high-contrast and unoccluded, with the
estimate landing on a diver's gear. Distance was a proxy: boards held close
are presented at more varied angles, so they fall outside a half sweep more
often.

| metric (52-frame subset) | 0–180° | **0–360°** |
| --- | ---: | ---: |
| frames seeded at ECC ≥ 0.80 | 30 (58%) | **37 (71%)** |
| median px above gate | 6.1 | **6.0** |
| p90 px above gate | 7.6 | **7.5** |

Coverage rises while accuracy above the gate is unchanged — these were frames
the gate correctly rejected, now found rather than fits improved.

**V-Slate 3 was entirely a casualty of this**, which retracts an earlier
conclusion that it failed for board-specific reasons:

| V-Slate 3 | 0–180° | 0–360° |
| --- | ---: | ---: |
| median offset | 1607 mm | **55.9 mm** |
| median angle | 63.5° | **11.95°** |
| median ECC | 0.686 | **0.909** |

Cost is 2× the search, which is affordable. The lesson worth keeping: a
symmetry assumption imported from regular calibration targets is invalid for
deliberately irregular ones, and it survived four rounds of unrelated tuning
because every metric it broke looked like an accuracy problem.

#### The whole-page template patch — second structural bug

`template_pattern_quad` took the bounding box of *all dark pixels* in the
render. Every template has a page border, so the "pattern" came out as **100%
of the page** on all eleven templates. The patch the search matched against was
therefore ~78% white margin for the V-Slates and 3× oversized for the grids —
so little signal that a thin dark object (a diver's speargun, in the failure
that exposed it) could outscore an obvious, well-lit board.

Fixed by deriving the patch from the **reference points**, which are by
definition the fiducial pattern, plus padding for stroke width.

| full corpus | before | **after** |
| --- | ---: | ---: |
| seeded at ECC ≥ 0.80 | 62/104 (60%) | **70/104 (67%)** |
| median px above gate | 5.7 | 5.7 |
| median offset (all) | 103 mm | **78 mm** |

Per family after the fix:

| slate | seeded | med px | med offset (gated) | med angle | ECC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tic-Tac-Toe 6 | 83% | 5.1 | **8.2 mm** | **2.07°** | 0.955 |
| V-Slate 1 | 73% | 5.7 | 58.2 mm | 30.2° | 0.882 |
| V-Slate 3 | 56% | 7.5 | 50.9 mm | 23.9° | 0.894 |
| V-Slate 4 | 44% | 5.7 | **27.1 mm** | 47.1° | 0.699 |

**Tic-Tac-Toe went from worst to best** (45.3° → 2.07°), because its patch was
the most oversized. That retracts the earlier decision to exclude the grid
family from seeding: the "confident and wrong at 81 px" behaviour was this bug,
not the pattern. `SUPPORTED_FAMILIES` now includes it. H-Slate stays out — it
has *no* labeled frames, and zero evidence is not the same as good evidence.

It also explains the Tic-Tac-Toe regression under the 360° sweep that nearly
justified a symmetry-aware sweep mechanism. Reverting that was correct: the
cause was the patch, and the mechanism would have been a wrong fix cemented on
top of a real bug.

**The remaining problem is coverage, not accuracy.** Gated offsets are now
27–51 mm across every family; V-Slate 4 has the *best* gated offset of any
V-Slate and merely reaches it only 44% of the time.

#### The learned board mask (Phase 2)

Reached only after tuning was exhausted, and scoped to the one thing tuning
provably could not fix. The classical geometry was already good — gated plane
offsets 27–58 mm, 5.7 px points — while *localization* failed, because the
search must scan the whole frame and reef competes with the board. Adding
search hypotheses made that **worse** (finer scale grid: 67% → 63%), since
every extra candidate is another chance for background to win the max. A mask
removes the competition instead of out-tuning it.

**Training data was free.** The human labels give correspondences, so a
homography warps the template's page outline into each frame — that outline
*is* the board. 104 frames → 104 exact masks, no new annotation.

Small UNet (1.08M params), Dice+BCE (the board is a median **1.1%** of frame,
so plain BCE is minimised by predicting all-background), split **by dive**.

| leave-one-dive-out | final hit |
| --- | ---: |
| 341 / 347 / 349 / 466 / 471 | 1.00 |
| 383 (V-Slate 3, sole dive) | 0.89 |
| 465 | 0.80 |
| 279 (Tic-Tac-Toe, sole dive) | 0.67 |
| **weighted** | **0.95** |

Integrated as *additional* candidates alongside the classical ones, so a wrong
or empty mask costs only search time — never coverage. Evaluated on
**out-of-fold** masks: every frame's mask comes from a model that never saw its
dive.

| full corpus | classical | **+ mask** |
| --- | ---: | ---: |
| gated coverage | 67% | **80%** |
| median px above gate | 5.7 | 5.9 |
| median plane offset | 78.4 mm | **62.0 mm** |
| V-Slate 1 seeded | 73% | **89%** |
| V-Slate 4 seeded | 44% | **56%** |

V-Slate 4 gained most in quality (ECC 0.699 → 0.883, angle 47° → 24°) — the
board that resisted four rounds of search tuning, and one the model never saw
in training.

#### Two evaluation errors worth not repeating

Both would have concluded *against* the learned model, and both were mine.

1. **An over-pessimistic split.** Holding out three dives at once (to "span
   three families") removed *every* Tic-Tac-Toe and V-Slate 3 frame from
   training, so the model was scored on board types it had never seen, on 77
   frames instead of 95. That produced 0.52 hit rate and an apparent
   overfitting problem. Leave-one-dive-out gives **0.95** and no overfitting.
   The lesson: with 8 dives and one dominant board type, a multi-dive holdout
   silently becomes a zero-shot test.
2. **An inert integration.** The first integrated run returned results
   *bit-identical* to classical-only — tempting to read as "the mask doesn't
   help". It was a units bug: the seed maps the **pattern** extent onto a
   candidate, but the mask outlines the **whole board**, so the pattern was
   stretched across the page, scored poorly, and never survived to ECC
   refinement. Bit-identical output was the tell — an unhelpful candidate would
   still perturb *something* across 104 frames; perfect identity means the code
   path never changed.

#### Method note

Three iterations, three structural bugs — panel offset, 180° sweep, whole-page
patch — each worth more than every tuning attempt combined. All four careful
refinement attempts (coarse-to-fine ECC, pre-ECC normalization, full-resolution
crop, template blur) produced **zero** net improvement. Every real gain came
from rendering the failures and looking at where the estimate actually landed;
twice the summary statistics pointed at a plausible and wrong mechanism
(obliquity; "V-Slate 4 is a hard board").

#### Pre-annotation quality is a *different* axis — and it is already good

For calibration the points are an arbitrary parameterization, so per-point
pixel error is the wrong metric. For a Label Studio pre-annotation it is the
*only* metric that matters: the labeler's work is dragging dots.

Measured over 52 frames (median per-point distance from the human label):

| gate | frames kept | med px | p75 | p90 | med plane offset |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 52 (100%) | 7.4 | 341.2 | **1262.1** | 116 mm |
| **ECC ≥ 0.80** | 30 (58%) | **6.1** | **6.7** | **7.6** | 51 mm |
| ECC ≥ 0.90 | 5 (10%) | 3.3 | 5.1 | 5.7 | 80 mm |

Above the gate the distribution is extraordinarily tight — median 6.1 px, p90
7.6 px, on a board spanning ~300 px. The ungated p90 of 1262 px is pure
failure-tail, and the gate removes all of it.

**Pixel error and plane error are uncorrelated above the gate** (rank
correlation **+0.003**). They are genuinely independent quality axes: 2-D
alignment can be excellent while absolute depth — which sets the plane offset
— is still a few percent off, because depth is weakly determined by a small
planar target. A prediction can therefore be *good for labeling and mediocre
for calibration at the same time*, which is exactly the situation here.

Per slate at ECC ≥ 0.80:

| slate | kept | med px | p90 px |
| --- | ---: | ---: | ---: |
| V-Slate 1 | 25 | **6.0** | 7.4 |
| V-Slate 4 | 3 | **6.2** | 6.8 |
| Tic-Tac-Toe 6 | 2 | **80.8** | 141.4 |
| V-Slate 3 | 0 | — | — |

The V-Slates are ready. **Tic-Tac-Toe passes the gate while being badly
wrong** — the ECC threshold is not calibrated for it (grid patterns correlate
well even when misaligned by a whole cell), so it must be excluded or
separately thresholded before any of this is enabled. Two frames, but the
failure direction is the dangerous one: confident and wrong.

#### What this establishes

`local_std` separates board from **water**, not board from **reef** — a
semantic distinction, not a statistical one, and the boundary where hand-tuned
thresholds stop. The template search sidesteps it by never segmenting at all.

Residual error is dominated by **tilt** (~20°) while position is good (65 mm),
the signature of a weakly-constrained plane normal. Expect this to be
template-dependent: a 6-vertex chevron is a thin shape whose silhouette barely
changes under moderate out-of-plane rotation, whereas eight scattered tape
corners should constrain the normal far better.

#### Board design context (from the maintainer)

The slates are duct tape stuck on dive slates by hand, and **haphazard by
design**. That is a deliberate and sound choice: unique irregular tape makes
orientation unambiguous, lets each board identify itself, and conditions PnP
*better* than a regular grid, which aliases against its own symmetry.

Two consequences: the per-board template is load-bearing (six Tic-Tac-Toe
templates exist because there are six physically different boards, and their
0.2–1.2% deviation from 180° symmetry *is* the tape irregularity, not noise);
and orientation genuinely matters at labeling time, since a flipped assignment
pairs measured corner 1 against physical corner 8.

**The template PDFs are the measurements** — authoritative by definition. So
`reference_points / dpi → inches → metres` is exact ground truth and there is
no template-vs-reality error term to chase.

**Recommendation:** classical is now viable enough to be the real baseline, but
not yet accurate enough to ship. Next lever is §3 option 3 — learn only the
board/pattern mask and keep all geometry classical — which should sharpen
precisely the boundary the tilt estimate is starving for. Everything downstream
of the mask is already built and tested.

Two caveats the renders surface:

- **The Tic-Tac-Toe boards are hand-drawn.** That is why six separate
  templates exist — each physical board has its own measured geometry, and
  strokes are irregular and non-parallel. A generic "tic-tac-toe detector"
  won't transfer between them; matching must be against the specific
  `dive_slate_id`'s template. Fortunately that ID is always known.
- **Scale is small.** On a 4014×3016 frame the board occupies roughly
  300–800 px across (V-Slates smaller, grids larger). Corner precision at that
  scale is what the ~0.64 px human noise floor has to be met against, so
  subpixel refinement is not optional.

---

## 4. Evaluation

Because the points are an arbitrary parameterization (§3), **per-point pixel
error against the human clicks is not the metric** — a prediction can differ
from the human labeling point-for-point and still produce an identical pose.
Agreement with the human *calibration* is what counts:

0. **Per-frame plane agreement** — the primary per-frame quantity, and the
   only part of a prediction that reaches the calibration:
   `geometry.plane_difference` → **(normal angle in degrees, offset in mm)**
   against the human-derived plane. Parameterization-independent, and blind to
   the in-plane freedoms the calibration ignores — so it neither rewards nor
   penalizes matching the human's arbitrary point choices.
1. **Per-frame reprojection error** — predicted points → `solvePnP` →
   reproject template → RMS pixel residual. Cheap, needs no oracle, and doubles
   as the confidence gate. But note it measures *self*-consistency: a
   confidently wrong pose can still reproject cleanly (see §1.5, where exactly
   that happened), so it gates but does not validate.
2. **Per-dive extrinsics agreement** — recompute `LaserExtrinsics` from
   predicted labels; compare `laser_position` / `laser_axis` against the
   human-derived row. The stage-13 refactor was validated to 0.011° / 0.39 mm,
   which is a reasonable order-of-magnitude target for the accept bar.
3. **End-to-end measurement agreement** — stage 14 fish measurements under
   predicted vs. human calibration.

Report per slate type. Held-out split must be **by dive**, not by frame —
frames within a dive share a board, a camera, and lighting, so a frame-level
split leaks badly. With exactly 8 dives that means 8 leave-one-dive-out folds,
only 5 of them V-Slate 1, and dive 341 alone holding a third of the corpus.
Headline numbers will carry wide error bars; report per-dive and treat any
single fold as anecdote.

**The gate threshold is measured, not guessed.** Running stage 13's own PnP
over the 104 human labels, *after* the §1.5 repair, gives a median RMS
reprojection of **0.64 px**, p90 1.22 px, max 3.11 px, zero PnP failures
(`scripts/check_pnp_residuals.py`). So a prediction whose residual exceeds
~3.5 px is less self-consistent than any human label in the corpus and should
never auto-accept. Passing the gate is necessary, not sufficient — metric 0
and 2 decide correctness.

That same run confirms human click order is sound: a swapped or mis-ordered
click on a board this size yields residuals in the hundreds of pixels, and the
worst label in the corpus is 3.11 px. (For predictions this question is moot —
ordering comes from `H` by construction.)

Confidence gate: reprojection residual + count of RANSAC inliers. Below
threshold → defer to human, never auto-calibrate.

---

## 5. Data access — unblocked

Both halves are now in hand:

- **Metadata** from the production dump `2026-07-31T03-00-01Z.dump`, restored
  locally: labels, template geometry, intrinsics, laser labels, and all 8 human
  calibrations.
- **Pixels** from the NAS at `~/mnt/fishsense_data/REEF/data`. **104/104 frames
  and 11/11 template PDFs resolve** against the paths stored in the DB, so
  Garage S3 is not needed.

Raw `.ORF` decode + `cv2.undistort` reproduces the labeling space
(`scripts/verify_label_overlay.py`). Note this uses `rawpy` rather than
fishsense-core's Rust raw path — identical geometry, different tone curve. If
byte-parity with the production rectification ever matters, swap in
`fishsense_core.image.RectifiedImage`.

---

## 6. Open questions

- ~~Are `reference_points` reliably in template order?~~ **Resolved** — §4's
  residual run shows a 3.11 px worst case, orders of magnitude below what a
  mis-ordered click would produce. Ordering is sound.
- ~~Do multi-skip labels exist in quantity?~~ **Resolved** — none exist; §2's
  bug is latent.
- ~~Which dives have both slate labels and a human `LaserExtrinsics`?~~
  **Resolved** — all 8.
- ~~Are images reachable?~~ **Resolved** — 104/104 frames and 11/11 templates
  on the NAS.
- **Should the existing `LaserExtrinsics` be recomputed after the §1.5 fix?**
  They were all fitted from composite-space points. The measured impact is
  small (a few mm of laser-ray collinearity) but systematic, and they are the
  eval oracle. Recomputing is cheap and makes the oracle trustworthy — this is
  the main decision blocking Phase 1.
- **Can the seven unlabeled templates be validated at all?** Synthetic pretrain
  is the only path to them, and there is no real data to check it against.
  Worth deciding whether they are in scope for v1 or explicitly deferred to
  human labeling.
- **Is V-Slate 3's 1.42 px median a board property or a dive property?** It is
  the worst per-type median but comes from a single dive (383), so the two are
  confounded. Only more data separates them.
