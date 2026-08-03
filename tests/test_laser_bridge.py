"""Tests for the laser-geometry checks.

Two questions, same machinery, neither needing slate labels:

* **Collinearity** — do a dive's estimated planes agree on *some* laser ray?
  Validates a calibration from within the dive.
* **Bridge** — do they agree with *this specific* candidate ray, taken from
  another dive? Confirms (or refuses) reusing one dive's extrinsics on another.

The bridge test matters because mounts are PLA and drift between deployments,
so a shared `camera_id` is evidence of a shared camera body, not a shared
laser-to-camera transform. Each dive is an island until measurement says
otherwise.
"""

import numpy as np
import pytest

from fishsense_core.plane import Plane

from slate_training.laser_bridge import (
    bridge_residuals,
    line_residual_mm,
    predict_laser_pixel,
)

K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]])


class TestPredictLaserPixel:
    def test_axis_aligned_ray_hits_plane_at_principal_point(self):
        # Laser at the camera origin firing straight ahead, board at z=2.
        px = predict_laser_pixel(
            Plane(np.array([0.0, 0.0, 1.0]), 2.0),
            np.zeros(3), np.array([0.0, 0.0, 1.0]), K,
        )
        assert px == pytest.approx((500.0, 400.0))

    def test_offset_laser_projects_off_centre(self):
        # Laser mounted 10 cm to the right, still firing forward.
        px = predict_laser_pixel(
            Plane(np.array([0.0, 0.0, 1.0]), 2.0),
            np.array([0.10, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), K,
        )
        # x = 0.10 m at z = 2 m, f = 1000 -> 50 px right of centre.
        assert px[0] == pytest.approx(550.0)
        assert px[1] == pytest.approx(400.0)

    def test_closer_board_moves_an_angled_ray(self):
        """The whole basis of the check: where the dot lands depends on depth."""
        origin, axis = np.array([0.10, 0.0, 0.0]), np.array([-0.05, 0.0, 1.0])
        near = predict_laser_pixel(Plane(np.array([0.0, 0.0, 1.0]), 1.0), origin, axis, K)
        far = predict_laser_pixel(Plane(np.array([0.0, 0.0, 1.0]), 3.0), origin, axis, K)
        assert near[0] != pytest.approx(far[0])

    def test_ray_parallel_to_plane_returns_none(self):
        assert predict_laser_pixel(
            Plane(np.array([0.0, 0.0, 1.0]), 2.0),
            np.zeros(3), np.array([1.0, 0.0, 0.0]), K,
        ) is None

    def test_intersection_behind_the_camera_returns_none(self):
        assert predict_laser_pixel(
            Plane(np.array([0.0, 0.0, 1.0]), -2.0),
            np.zeros(3), np.array([0.0, 0.0, 1.0]), K,
        ) is None


class TestBridgeResiduals:
    def _planes(self, depths):
        return [Plane(np.array([0.0, 0.0, 1.0]), d) for d in depths]

    def test_correct_extrinsics_give_near_zero_residuals(self):
        origin, axis = np.array([0.10, 0.0, 0.0]), np.array([-0.04, 0.01, 1.0])
        planes = self._planes([1.2, 1.8, 2.4, 3.0])
        observed = [predict_laser_pixel(p, origin, axis, K) for p in planes]
        res = bridge_residuals(planes, observed, origin, axis, K)
        assert np.max(res) == pytest.approx(0.0, abs=1e-6)

    def test_a_shifted_mount_produces_large_residuals(self):
        """A PLA mount that moved 1 cm should be plainly visible."""
        origin, axis = np.array([0.10, 0.0, 0.0]), np.array([-0.04, 0.01, 1.0])
        planes = self._planes([1.2, 1.8, 2.4, 3.0])
        observed = [predict_laser_pixel(p, origin, axis, K) for p in planes]
        moved = origin + np.array([0.01, 0.0, 0.0])
        res = bridge_residuals(planes, observed, moved, axis, K)
        assert np.median(res) > 3.0

    def test_residuals_are_positive_and_per_frame(self):
        planes = self._planes([1.5, 2.5])
        origin, axis = np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        observed = [predict_laser_pixel(p, origin, axis, K) for p in planes]
        res = bridge_residuals(planes, observed, origin, axis, K)
        assert len(res) == 2 and (res >= 0).all()

    def test_unusable_frames_are_dropped_not_counted_as_zero(self):
        # A plane the ray cannot hit must not silently score a perfect match.
        planes = [Plane(np.array([1.0, 0.0, 0.0]), 2.0)]   # edge-on
        res = bridge_residuals(planes, [(500.0, 400.0)],
                               np.zeros(3), np.array([0.0, 0.0, 1.0]), K)
        assert len(res) == 0


class TestLineResidual:
    def test_perfectly_collinear_points_score_zero(self):
        pts = [[0, 0, 1.0], [0.01, 0, 1.5], [0.02, 0, 2.0], [0.03, 0, 2.5]]
        assert line_residual_mm(pts) == pytest.approx(0.0, abs=1e-9)

    def test_scatter_is_reported_in_mm(self):
        pts = np.array([[0, 0, 1.0], [0, 0, 2.0], [0, 0, 3.0], [0, 0.01, 2.0]])
        assert line_residual_mm(pts) > 1.0

    def test_uniform_scaling_about_the_camera_preserves_collinearity(self):
        """The blind spot worth documenting: a depth-scale error survives this
        check, because scaling about the origin maps a line to a line. That is
        why the rig-plausibility band is needed alongside it."""
        pts = np.array([[0.1, 0, 1.0], [0.1, 0, 2.0], [0.1, 0, 3.0]])
        assert line_residual_mm(pts * 1.5) == pytest.approx(
            line_residual_mm(pts), abs=1e-9)
