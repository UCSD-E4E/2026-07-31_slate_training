"""Tests for the plane target — what the calibration actually consumes.

Stage 13 uses only `R[:,2]` (the board normal) and one point on the board to
intersect the laser ray. Verified numerically: in-plane rotation (any angle,
including 180 deg) and in-plane translation change the resulting laser point
by exactly 0 mm; only motion along the normal matters.

So the estimator's target is a plane `n . X = d` in camera space, 3 DOF.
These tests pin that invariance down so it can't silently regress.
"""

import numpy as np
import pytest

from slate_training.geometry import (
    Plane,
    laser_point_on_plane,
    plane_difference,
    plane_from_pose,
)


def _rz(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0],
                     [np.sin(t), np.cos(t), 0.0],
                     [0.0, 0.0, 1.0]])


class TestPlaneFromPose:
    def test_identity_pose_gives_z_normal_at_translation_depth(self):
        plane = plane_from_pose(np.eye(3), np.array([0.0, 0.0, 2.0]))
        assert np.allclose(plane.normal, [0.0, 0.0, 1.0])
        assert plane.distance == pytest.approx(2.0)

    def test_normal_is_unit_length(self):
        rot = _rz(30.0) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        plane = plane_from_pose(rot, np.array([0.1, 0.2, 1.5]))
        assert np.linalg.norm(plane.normal) == pytest.approx(1.0)

    @pytest.mark.parametrize("deg", [17.0, 90.0, 180.0, 359.0])
    def test_in_plane_rotation_leaves_the_plane_unchanged(self, deg):
        """The invariance that makes `upside_down` irrelevant to calibration."""
        rot = np.eye(3)
        t = np.array([0.05, -0.1, 2.0])
        base = plane_from_pose(rot, t)
        spun = plane_from_pose(rot @ _rz(deg), t)
        angle, dist_mm = plane_difference(base, spun)
        assert angle == pytest.approx(0.0, abs=1e-9)
        assert dist_mm == pytest.approx(0.0, abs=1e-9)

    def test_in_plane_translation_leaves_the_plane_unchanged(self):
        rot = np.eye(3)
        base = plane_from_pose(rot, np.array([0.0, 0.0, 2.0]))
        # rot[:,0] and rot[:,1] span the board plane; sliding along them is free.
        slid = plane_from_pose(rot, np.array([0.5, -0.3, 2.0]))
        angle, dist_mm = plane_difference(base, slid)
        assert angle == pytest.approx(0.0, abs=1e-9)
        assert dist_mm == pytest.approx(0.0, abs=1e-9)

    def test_motion_along_the_normal_does_change_the_plane(self):
        base = plane_from_pose(np.eye(3), np.array([0.0, 0.0, 2.0]))
        moved = plane_from_pose(np.eye(3), np.array([0.0, 0.0, 2.01]))
        angle, dist_mm = plane_difference(base, moved)
        assert angle == pytest.approx(0.0, abs=1e-9)
        assert dist_mm == pytest.approx(10.0, abs=1e-6)


class TestPlaneDifference:
    def test_reports_normal_angle_in_degrees(self):
        a = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
        b = Plane(np.array([0.0, np.sin(np.radians(5)), np.cos(np.radians(5))]), 2.0)
        angle, _ = plane_difference(a, b)
        assert angle == pytest.approx(5.0, abs=1e-6)

    def test_sign_flipped_normal_is_the_same_plane(self):
        # A normal and its negation describe one plane; the metric must not
        # report 180 deg for what is physically identical.
        a = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
        b = Plane(np.array([0.0, 0.0, -1.0]), -2.0)
        angle, dist_mm = plane_difference(a, b)
        assert angle == pytest.approx(0.0, abs=1e-9)
        assert dist_mm == pytest.approx(0.0, abs=1e-9)


class TestLaserPointOnPlane:
    def test_ray_hits_the_plane_at_the_expected_depth(self):
        # Camera at origin, board 2 m away facing the camera, laser at the
        # principal point -> the hit must be straight ahead at z = 2.
        K = np.array([[1000.0, 0, 500.0], [0, 1000.0, 400.0], [0, 0, 1.0]])
        plane = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
        point = laser_point_on_plane(plane, (500.0, 400.0), K)
        assert np.allclose(point, [0.0, 0.0, 2.0], atol=1e-9)

    def test_off_axis_ray_lands_off_axis_on_the_plane(self):
        K = np.array([[1000.0, 0, 500.0], [0, 1000.0, 400.0], [0, 0, 1.0]])
        plane = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
        point = laser_point_on_plane(plane, (600.0, 400.0), K)
        # 100 px off centre at f=1000 and 2 m -> 0.2 m lateral.
        assert point[0] == pytest.approx(0.2, abs=1e-9)
        assert point[2] == pytest.approx(2.0, abs=1e-9)

    def test_ray_parallel_to_the_plane_returns_none(self):
        K = np.array([[1000.0, 0, 500.0], [0, 1000.0, 400.0], [0, 0, 1.0]])
        plane = Plane(np.array([1.0, 0.0, 0.0]), 2.0)  # plane edge-on
        assert laser_point_on_plane(plane, (500.0, 400.0), K) is None
