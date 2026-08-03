"""Label-free geometric checks on a dive's calibration.

Reprojecting slate labels does **not** validate a calibration. A planar target
always admits an exact homography, so a low reprojection residual measures
self-consistency, not correctness — demonstrated on this project's own data,
where 104 labels sitting 4968 px in the wrong coordinate space still reprojected
at 1.06 px median. The labels are not ground truth either: human clicks carry
~0.64 px of noise and the template is itself a measurement of hand-placed tape.

The laser is an external physical constraint, and it needs no slate labels:

* :func:`line_residual_mm` — **collinearity.** Intersect each frame's labelled
  laser pixel with that frame's estimated board plane; the 3-D points must lie
  on one line. N independently-estimated planes constrained to agree on two line
  parameters, so uncorrelated plane error scatters immediately.

* :func:`bridge_residuals` — **bridge confirmation.** Given a *candidate*
  extrinsics from another dive, predict where the laser dot should appear and
  compare against the labelled pixel. Mounts are PLA and drift between
  deployments, so a shared ``camera_id`` means a shared camera body, not a
  shared laser-to-camera transform: every dive is an island until measurement
  says otherwise.

Two properties worth keeping in mind.

**Collinearity is asymmetric evidence.** A small residual proves the planes are
right *and* the mount was stable, since neither can be bad without inflating it.
A large residual proves little — it cannot separate bad fits from a mount that
flexed mid-dive.

**Collinearity is blind to a uniform depth-scale error**, because scaling about
the camera origin maps a line to a line. Pair it with a rig-plausibility band on
the fitted line's closest approach to the camera (~10 cm here) to catch that.
Treat the band as coarse: with PLA mounts it rules out gross scale error, it
does not certify precision.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from fishsense_core.plane import Plane

__all__ = ["predict_laser_pixel", "bridge_residuals", "line_residual_mm",
           "closest_approach_m"]

Pixel = Tuple[float, float]


def predict_laser_pixel(
    plane: Plane,
    laser_origin: np.ndarray,
    laser_axis: np.ndarray,
    camera_matrix: np.ndarray,
) -> Pixel | None:
    """Where a candidate laser ray should appear, given a board plane.

    Ray ``X(t) = origin + t * axis`` meets ``n · X = d`` at
    ``t = (d - n·origin) / (n·axis)``. Returns None when the ray is parallel to
    the plane, meets it behind the emitter, or lands behind the camera — those
    frames carry no information and must be dropped rather than scored.
    """
    normal = np.asarray(plane.normal, dtype=float)
    origin = np.asarray(laser_origin, dtype=float).reshape(3)
    axis = np.asarray(laser_axis, dtype=float).reshape(3)

    denominator = float(normal @ axis)
    if abs(denominator) < 1e-12:
        return None

    t = (float(plane.distance) - float(normal @ origin)) / denominator
    if t <= 0:
        return None

    point = origin + t * axis
    if point[2] <= 1e-9 or not np.isfinite(point).all():
        return None

    projected = np.asarray(camera_matrix, dtype=float) @ point
    return (float(projected[0] / projected[2]), float(projected[1] / projected[2]))


def bridge_residuals(
    planes: Sequence[Plane],
    observed_pixels: Sequence[Pixel | None],
    laser_origin: np.ndarray,
    laser_axis: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Per-frame pixel error of a candidate extrinsics on a target dive.

    Frames the ray cannot reach are **dropped**, not scored as zero — counting
    them as perfect agreement is how a candidate that never intersects anything
    would look ideal.

    A mount that shifted between deployments shows up as a coherent bias rather
    than scatter, so the sign and structure of the residuals say *how* it moved,
    not merely that it did.
    """
    out = []
    for plane, observed in zip(planes, observed_pixels):
        if observed is None:
            continue
        predicted = predict_laser_pixel(plane, laser_origin, laser_axis, camera_matrix)
        if predicted is None:
            continue
        out.append(float(np.hypot(predicted[0] - observed[0],
                                  predicted[1] - observed[1])))
    return np.asarray(out, dtype=float)


def line_residual_mm(points: Iterable[Sequence[float]]) -> float:
    """RMS distance of 3-D points from their best-fit line, in millimetres."""
    array = np.asarray(list(points), dtype=float)
    if len(array) < 3:
        return float("nan")
    singular = np.linalg.svd(array - array.mean(axis=0), compute_uv=False)
    return float(np.sqrt((singular[1] ** 2 + singular[2] ** 2) / len(array))) * 1000.0


def closest_approach_m(points: Iterable[Sequence[float]]) -> float:
    """How near the fitted laser line passes to the camera origin, in metres.

    The rig-plausibility band. Catches the uniform depth-scale error that
    collinearity cannot see.
    """
    array = np.asarray(list(points), dtype=float)
    if len(array) < 2:
        return float("nan")
    centre = array.mean(axis=0)
    direction = np.linalg.svd(array - centre)[2][0]
    direction = direction / np.linalg.norm(direction)
    return float(np.linalg.norm(centre - (centre @ direction) * direction))
