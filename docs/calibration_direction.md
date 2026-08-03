# Calibration direction — flat port, Pinax, and label-free validation

Design notes from the 2026-08-02 discussion. Nothing here is built except the
label-free checks in `laser_bridge.py`; the rest is decisions and open
questions recorded so they aren't re-derived.

---

## 1. Everything measured so far is dome-port

All ten dives analysed in this repo — the 8 reef dives and pool dives 65/71 —
are **dome port**. Every number in [design.md](design.md) and
[data_report.md](data_report.md) is dome-port data.

**None of it transfers to flat.** The straight laser lines, the "distortion
coefficients are sound" conclusion, the ECC ≥ 0.80 threshold, the 5.9 px
pre-annotation accuracy: all conditioned on a central projection model that a
flat port breaks. Stated explicitly because those numbers read like general
validation.

## 2. Flat port breaks the slate estimator — and Pinax fixes it

The estimator rests on a chain whose first link a flat port severs:

> planar board → **homography** → ECC alignment → `solvePnP` → plane

Under refraction a plane does not image via a homography, so the error would be
systematic and grow with field angle and depth — not random noise.

[fishsense-pinax](https://github.com/UCSD-E4E/fishsense-pinax) resolves this.
Near an optimal camera-to-glass distance `d0*` the axial camera collapses to a
**virtual pinhole**, so a per-pixel correction map rectifies flat-port images
back to something the existing geometry consumes unchanged. Calibration happens
**in air**; water enters only through the refraction index.

Consequences for this pipeline:

* Rectification must slot in where `RectifiedImage` currently does
  `cv2.undistort`. Downstream (`fishsense_core.slate`, stage 13) is unchanged.
* **Build the housing at `d0*`.** The pinhole collapse is only accurate near
  it; `pinax.optimize_d0` computes it.
* `mapgen` emits a **validity mask**. The template search must respect it
  rather than treat masked pixels as image content.

## 3. Laser self-calibration — possible, but fighting `d0*`

### What laser labels alone give you (measured, dome)

The image of a 3-D line is a 2-D line, so every laser dot on a dive must be
collinear in the **undistorted** image — regardless of depth or surface, and
**with no board in frame**. Measured across 10 dives: residuals of 1–3 px over
165–557 px of spread, with no detectable curvature (adding a quadratic term
changes the residual by <0.2 px).

That line determines the plane Π through the camera centre containing the laser
ray — **2 of the ray's 4 DOF**. Validated against all six production
calibrations: their fitted rays lie in Π to **0.2–2.2 mm** and **0.02–0.08°**.

Two immediate uses, both needing no slate labels:

* **A per-dive QC signal with far more data than the fit uses.** Stage 13 fits
  4 DOF from only the paired slate+laser frames — 5 to 33 of them — while dive
  341 has **266** laser observations. The other 233 are discarded. Constraining
  the ray to Π and solving only the remaining 2 DOF from the paired frames
  would lower variance on every calibration, and helps thin dives most.
* **A partial bridge test.** A candidate ray from another dive must lie in Π.
  Catches a mount that moved out of plane; blind to one that slid within it.

### Why the last 2 DOF need depth

Central projection is scale-invariant: scaling a point about the camera centre
leaves its image unchanged, so depth is unrecoverable and every line in Π
images identically.

**Refraction breaks that invariance**, because the interface introduces an
absolute length scale (pupil-to-glass distance). Two rays indistinguishable
under a pinhole model image as *different curves* through a flat port — so the
curvature encodes the missing DOF, and full 4-DOF recovery from dots alone
becomes possible in principle. Laser calibration with no slate at all.

### The tension, and its resolution

`optimize_d0` chooses `d0*` to *minimise* the non-central behaviour — precisely
the signal self-calibration would exploit. Optimising the housing for Pinax
suppresses it.

**Resolution: rectification is post-processing, not optics.** Build at `d0*`,
run the slate/measurement pipeline on rectified images, and run
self-calibration on the **distorted originals**, where the curvature is still
physically present. The `.ORF` files are the originals; nothing is lost.

This also yields a genuinely independent cross-check — self-calibration on
distorted pixels versus slate PnP on rectified ones use different images,
different geometry and different failure modes. Agreement is strong evidence;
disagreement localises the fault. The pool incident happened because ECC was
the *only* check.

**Open question:** at `d0*` the residual curvature is small by construction.
Is it invertible against the observed 1–3 px label noise? A simulation answers
this without hardware: take a known ray from a calibrated dive, forward-model
its dots through a flat port at `d0*`, add realistic noise, attempt recovery.

**Note the same idea may apply to the existing dome corpus.** Real domes are
never perfectly centred, so residual aberration may carry the same information
— testable today against six known-good rays.

## 4. Calibration targets — LEGO, no printed patterns

Printed chessboards/ChArUco are rejected as not citizen-science accessible,
consistent with the duct-tape slates.

LEGO is a better substrate than a compromise:

* **Stud pitch 8.0 mm**, held to a few microns. A printed target's real error
  floor is printer scaling and paper stretch — a few tenths of a percent. A
  parts list is a globally reproducible spec; a PDF sent to an unknown printer
  is not.
* **Plate height 3.2 mm** gives exact quantised depth offsets. That 3-D
  structure is what breaks the planar-calibration degeneracy, where
  near-fronto-parallel views leave focal length, principal point and distortion
  poorly separated.

Design notes:

* **Build the pattern from contrasting 1×1 pieces on a baseplate.** Bare studs
  rely on shading, so detection becomes lighting- and viewpoint-dependent.
* **Square tiles over round plates.** Round gives a circle grid with built-in
  OpenCV support, but under perspective the *ellipse centroid is not the
  projection of the circle centre* — a bias growing with obliquity. Square
  tiles give projectively exact corner features.
* **Identity without markers:** asymmetric placement (unique alignment) plus a
  few colour-coded anchors. Same principle as the slates — deliberate
  irregularity as the identity certificate.
* **Rigidity beats accuracy.** Bundle-adjust the as-built geometry from LEGO's
  nominal dimensions as the initial guess. Stack-up tolerance becomes something
  estimated, not something that corrupts. Flex does not — brace it, keep it
  squat.
* **Frame-corner coverage drives distortion accuracy**, and failing it is the
  standard calibration mistake. It matters more with a flat port, where
  refraction error is largest at the periphery.

### In-air laser calibration

A stepped target puts the dot at many exactly-known depths: translate the rig
and it walks across steps of known height, each observation giving a 3-D point
on the ray with no translation measurement. Far better sampling than the 5–33
hand-held slate frames a dive currently provides. Use matte surfaces — glossy
ABS scatters the dot and biases the centroid straight into the ray estimate.

**The beam refracts on the way out too.** The laser-to-camera transform is
mechanical and medium-independent, but a flat port bends the exiting beam by
Snell's law. Given the port plane relative to the camera (which `d0*`
establishes), that correction is a deterministic single application of Snell —
the laser analogue of what Pinax does for the camera. It depends on salinity,
same as the map.

### The payoff

If laser extrinsics can be established in air pre-deployment, **slate labelling
drops out of the calibration path for new dives entirely** — a bigger lever
than automating it, which is what this repo does. The slate work remains
necessary for the historical corpus.

The objection is PLA mount drift: an in-air calibration is stale the moment the
mount moves. Which is what the §3 checks are for — **calibrate dry and precise,
verify wet and cheap**, per dive, from laser labels already collected. That also
satisfies island-by-default: each dive still gets its own confirmation, it just
no longer needs its own slate labelling to get one.

## 5. Schema gaps this exposes

`camera` holds only serial and name; `cameraintrinsics` holds only
`camera_matrix` and `distortion_coefficients`, keyed on `camera_id` with **no
validity period**. So:

* **A re-ported camera silently inherits its dome intrinsics.** Nothing errors;
  measurements just quietly become wrong. With a mixed fleet during the
  changeover this is the failure most likely to bite and the cheapest to
  prevent.
* **No record of projection model, water index, or correction map.** A dive's
  geometry is only reproducible if the exact map can be regenerated.
* **Raw laser pixel coordinates must be preserved.** Rectification destroys the
  curvature irreversibly. If labelling happens on rectified frames and only
  those coordinates are stored, self-calibration becomes impossible for every
  flat-port dive ever recorded — and nothing would look broken.

Same shape as the PLA-mount insight: extrinsics are per-dive because mounts
drift, but intrinsics are modelled as a permanent property of a camera body.
A port swap makes that false, and changes the *projection model*, not just its
parameters.

## 6. What validates a calibration (no slate labels needed)

Reprojecting slate labels does **not** validate a calibration — a planar target
always admits an exact homography. This project's own data proves it: 104
labels sitting 4968 px in the wrong coordinate space still reprojected at
1.06 px median.

| check | what it proves | blind to |
| --- | --- | --- |
| Laser-line planarity (Π) | ray lies in the right plane; validates distortion | position/direction within Π |
| Collinearity of 3-D laser points | planes are right *and* mount was stable | uniform depth-scale error |
| Fitted line ↔ camera distance | gross scale error | precision (PLA mounts vary) |
| Bridge residuals | a candidate ray is valid on this dive | needs board *and* laser in frame |
| End-to-end measurement | the whole chain | nothing — this is the acceptance test |

**Collinearity is asymmetric evidence.** A small residual proves planes and
mount are both good, since neither can be bad without inflating it. A large
residual proves little — it cannot separate bad fits from a mount that flexed
mid-dive.

Implemented in [`laser_bridge.py`](../src/slate_training/laser_bridge.py).
