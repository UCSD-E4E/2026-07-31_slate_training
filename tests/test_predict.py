"""Tests for the gated predict path — the unit a data-worker activity calls.

Every rejection rule here exists because of a measured failure; the reasons are
recorded so the worker can log *why* a frame was declined rather than silently
producing nothing.
"""

import numpy as np
import pytest

from slate_training.geometry import Plane
from slate_training.predict import (
    DEFAULT_MIN_CONFIDENCE,
    SUPPORTED_FAMILIES,
    SlatePrediction,
    slate_family,
    predict_slate,
)


class _Estimate:
    """Stand-in for BoardEstimate; predict_slate only reads these fields."""

    def __init__(self, ecc, n=6):
        self.ecc_score = ecc
        self.plane = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
        self.homography = np.eye(3)
        self.image_points = np.column_stack(
            [np.linspace(1500, 2100, n), np.linspace(1300, 1500, n)]
        )
        self.reprojection_rms = 0.8
        self.board_area_px = 50000.0


def _call(estimate, slate_name="V-Slate 1", **kw):
    return predict_slate(
        bgr=np.zeros((10, 10, 3), np.uint8),
        template_gray=np.zeros((10, 10), np.uint8),
        template_points=[(0.0, 0.0)] * 6,
        dpi=300,
        camera_matrix=np.eye(3),
        pdf_width_px=4200,
        pdf_height_px=2550,
        photo_width=4014,
        photo_height=3016,
        slate_name=slate_name,
        model_version="classical-v1",
        _estimator=lambda *a, **k: estimate,
        **kw,
    )


class TestSlateFamily:
    def test_v_slates_group_together(self):
        assert slate_family("V-Slate 1") == slate_family("V-Slate 4") == "v-slate"

    def test_tic_tac_toe_variants_group_together(self):
        assert slate_family("Tic-Tac-Toe 2") == slate_family("Tic-Tac-Toe 6")

    def test_h_slate_is_its_own_family(self):
        assert slate_family("H-Slate") == "h-slate"

    def test_v_slates_and_grids_are_supported_but_not_h_slate(self):
        # Tic-Tac-Toe was enabled once the template-patch bug was fixed: it
        # became the best family measured (83% seeded, 5.1px, 8.2mm).
        # H-Slate has *no* labeled frames, so it stays out on zero evidence.
        assert SUPPORTED_FAMILIES == frozenset({"v-slate", "tic-tac-toe"})
        assert "h-slate" not in SUPPORTED_FAMILIES


class TestGating:
    def test_confident_v_slate_yields_a_prediction(self):
        out = _call(_Estimate(0.89))
        assert isinstance(out, SlatePrediction)
        assert out.rejected_reason is None
        assert out.prediction is not None
        assert len(out.prediction["result"]) == 6
        assert out.confidence == pytest.approx(0.89)

    def test_low_confidence_is_declined(self):
        out = _call(_Estimate(0.62))
        assert out.prediction is None
        assert out.rejected_reason == "low_confidence"
        # The plane is still returned for telemetry even when not seeded.
        assert out.plane is not None

    def test_threshold_is_inclusive_at_the_boundary(self):
        assert _call(_Estimate(DEFAULT_MIN_CONFIDENCE)).prediction is not None

    def test_supported_grid_family_yields_a_prediction(self):
        assert _call(_Estimate(0.95), slate_name="Tic-Tac-Toe 6").prediction is not None

    def test_unsupported_family_is_declined_even_when_confident(self):
        out = _call(_Estimate(0.95), slate_name="H-Slate")
        assert out.prediction is None
        assert out.rejected_reason == "unsupported_slate_family"

    def test_no_board_found_is_declined(self):
        out = _call(None)
        assert out.prediction is None
        assert out.rejected_reason == "no_board"
        assert out.plane is None

    def test_threshold_is_overridable(self):
        out = _call(_Estimate(0.62), min_confidence=0.5)
        assert out.prediction is not None


class TestMaskIsOptional:
    """The learned mask must never be a hard dependency of the activity."""

    def test_works_without_a_mask(self):
        assert _call(_Estimate(0.89)).prediction is not None

    def test_mask_is_forwarded_to_the_estimator(self):
        seen = {}

        def estimator(*a, **kw):
            seen["mask"] = kw.get("board_mask")
            return _Estimate(0.89)

        predict_slate(
            bgr=np.zeros((10, 10, 3), np.uint8),
            template_gray=np.zeros((10, 10), np.uint8),
            template_points=[(0.0, 0.0)] * 6, dpi=300,
            camera_matrix=np.eye(3), pdf_width_px=4200, pdf_height_px=2550,
            photo_width=4014, photo_height=3016, slate_name="V-Slate 1",
            model_version="v1", board_mask=np.ones((4, 4), np.uint8),
            _estimator=estimator,
        )
        assert seen["mask"] is not None


class TestPredictionShape:
    def test_points_are_percentages_of_the_composite_canvas(self):
        out = _call(_Estimate(0.89))
        for item in out.prediction["result"]:
            assert item["original_width"] == 8981
            assert item["original_height"] == 3016
            assert 0.0 <= item["value"]["x"] <= 100.0
            assert 0.0 <= item["value"]["y"] <= 100.0

    def test_points_land_on_the_photo_half_of_the_canvas(self):
        # Panel is 4967 of 8981 px, i.e. 55.3%; a real board sits to the right.
        out = _call(_Estimate(0.89))
        assert all(i["value"]["x"] > 55.0 for i in out.prediction["result"])

    def test_score_and_version_are_carried(self):
        out = _call(_Estimate(0.91))
        assert out.prediction["score"] == pytest.approx(0.91)
        assert out.prediction["model_version"] == "classical-v1"
