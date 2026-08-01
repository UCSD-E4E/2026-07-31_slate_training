"""Tests for LS pre-annotation emission — the round trip back to labeling space."""

import pytest

from slate_training.contracts import composite_to_photo
from slate_training.label_studio import (
    build_prediction,
    composite_dimensions,
    points_to_keypoint_results,
)

# V-Slate 1 at 300 dpi against a 4014x3016 photo: the real production case.
VSLATE = dict(pdf_width_px=4200, pdf_height_px=2550, photo_width=4014, photo_height=3016)


class TestCompositeDimensions:
    def test_reproduces_the_observed_production_canvas(self):
        # Ground truth from a stored LS task: original_width == 8981.
        panel, width, height = composite_dimensions(**VSLATE)
        assert panel == 4967
        assert width == 8981
        assert height == 3016

    def test_truncates_rather_than_rounds(self):
        # The geometric width is 4968.0; stage 9's int() gives 4967. A rounded
        # implementation would silently shift every seeded point by a pixel.
        panel, _, _ = composite_dimensions(**VSLATE)
        assert panel != round(4200 * 3016 / 2550)

    def test_tic_tac_toe_has_a_different_panel(self):
        panel, width, _ = composite_dimensions(3300, 2550, 4014, 3016)
        assert panel == 3903
        assert width == 7917


class TestKeypointResults:
    def _dims(self):
        return composite_dimensions(**VSLATE)

    def test_photo_point_maps_to_composite_percentage(self):
        panel, width, height = self._dims()
        results = points_to_keypoint_results([(1000.0, 1508.0)], panel, width, height)
        assert len(results) == 1
        value = results[0]["value"]
        assert value["x"] == pytest.approx(100.0 * (1000.0 + panel) / width)
        assert value["y"] == pytest.approx(50.0)

    def test_declares_the_composite_dimensions(self):
        panel, width, height = self._dims()
        result = points_to_keypoint_results([(10.0, 10.0)], panel, width, height)[0]
        assert result["original_width"] == 8981
        assert result["original_height"] == 3016

    def test_round_trips_through_the_sync_shift(self):
        # What stage 12 will do to our prediction if a human accepts it.
        panel, width, height = self._dims()
        original = [(1234.5, 678.25)]
        value = points_to_keypoint_results(original, panel, width, height)[0]["value"]
        composite_x = value["x"] / 100.0 * width
        composite_y = value["y"] / 100.0 * height
        assert composite_to_photo([(composite_x, composite_y)], panel)[0] == (
            pytest.approx(original[0][0]),
            pytest.approx(original[0][1]),
        )

    def test_retains_template_index(self):
        panel, width, height = self._dims()
        results = points_to_keypoint_results(
            [(10.0, 10.0), (20.0, 20.0)], panel, width, height
        )
        assert [r["meta"]["template_index"] for r in results] == [0, 1]

    def test_out_of_canvas_points_are_dropped_not_clamped(self):
        panel, width, height = self._dims()
        results = points_to_keypoint_results(
            [(10.0, 10.0), (-99999.0, 10.0)], panel, width, height
        )
        assert len(results) == 1


class TestBuildPrediction:
    def _dims(self):
        return composite_dimensions(**VSLATE)

    def test_builds_a_prediction_with_score_and_version(self):
        panel, width, height = self._dims()
        pred = build_prediction(
            [(10.0, 10.0), (20.0, 20.0)], panel, width, height, 0.91, "classical-v1"
        )
        assert pred is not None
        assert pred["score"] == pytest.approx(0.91)
        assert pred["model_version"] == "classical-v1"
        assert len(pred["result"]) == 2

    def test_partial_point_set_yields_no_prediction(self):
        # A dropped point would break stage 13's positional pairing, and a
        # confident-looking partial annotation invites the labeler to accept it.
        panel, width, height = self._dims()
        assert build_prediction(
            [(10.0, 10.0), (-99999.0, 10.0)], panel, width, height, 0.9, "v1"
        ) is None

    def test_no_points_yields_no_prediction(self):
        panel, width, height = self._dims()
        assert build_prediction([], panel, width, height, 0.9, "v1") is None
