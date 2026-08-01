"""Pure-logic tests for the DiveSlateLabel coordinate/order contract.

These pin down the three facts the whole slate-detector design rests on,
all verified against fishsense-lite at 2026-07-31:

1. Label Studio shows the labeler a *composite* canvas
   ``[template render | rectified photo]``; stored `reference_points` /
   `slate_rectangle` have already had the panel width subtracted, so they
   live in rectified-photo pixels (same space as `LaserLabel.x/y`).
   -- sync_dive_slate_labels_for_label_studio_project_activity.py

2. `reference_points` is a *dense, ordered* list that positionally aligns
   with `DiveSlate.reference_points` after the `skipped_points` indices
   are removed. This is what stage 13 relies on to build solvePnP
   correspondences.
   -- perform_laser_calibration_activity._laser_point_in_camera_space

3. `slate_rectangle` is a 2-point axis-aligned bbox ``[(x0,y0),(x1,y1)]``
   from an LS RectangleLabels control -- NOT a 4-corner quad.
"""

import pytest

from slate_training.contracts import (
    composite_panel_width,
    drop_skipped,
    needs_panel_repair,
    photo_to_composite,
    repair_panel_offset,
    template_correspondences,
)


class TestCompositeOffset:
    """Panel width = pdf_aspect * photo_height; DPI cancels out."""

    def test_panel_width_scales_with_photo_height(self):
        # A 8.5x11 portrait slate against a 3000px-tall photo.
        assert composite_panel_width(8.5 / 11.0, 3000) == pytest.approx(2318.18, abs=0.01)

    def test_photo_to_composite_is_inverse_of_the_sync_shift(self):
        # sync stores (x - panel); seeding a prediction must add it back.
        assert photo_to_composite([(100.0, 50.0)], 2318.18) == [(2418.18, 50.0)]

    def test_photo_to_composite_leaves_y_untouched(self):
        assert photo_to_composite([(0.0, 7.5)], 999.0) == [(999.0, 7.5)]


class TestDropSkipped:
    """The subsetting stage 13 does -- but order-safe."""

    def test_no_skips_is_identity(self):
        pts = [(0, 0), (1, 1), (2, 2)]
        assert drop_skipped(pts, []) == pts

    def test_single_skip_removes_that_index(self):
        pts = [(0, 0), (1, 1), (2, 2)]
        assert drop_skipped(pts, [1]) == [(0, 0), (2, 2)]

    def test_multiple_skips_use_original_indices(self):
        # The regression that matters: stage 13 does a sequential
        # list.pop(idx), so after popping 2 the element originally at 5
        # has slid to 4 and popping 5 removes the WRONG point. Indices
        # must be interpreted against the original list.
        pts = [(i, i) for i in range(8)]
        assert drop_skipped(pts, [2, 5]) == [
            (0, 0), (1, 1), (3, 3), (4, 4), (6, 6), (7, 7)
        ]

    def test_unsorted_skips_behave_identically(self):
        pts = [(i, i) for i in range(8)]
        assert drop_skipped(pts, [5, 2]) == drop_skipped(pts, [2, 5])

    def test_out_of_range_index_is_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            drop_skipped([(0, 0), (1, 1)], [7])

    def test_duplicate_index_is_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            drop_skipped([(0, 0), (1, 1), (2, 2)], [1, 1])


class TestNeedsPanelRepair:
    """Detecting the historical composite-coordinate corpus.

    All 104 production labels were persisted WITHOUT stage 12's panel shift,
    so they sit in composite space. The tell is unambiguous: x beyond the
    photo width. Repair must be idempotent -- running it twice would push
    points off the left edge.
    """

    def test_composite_coords_are_detected(self):
        # V-Slate 1 on a 4014-wide photo: x ~6700 is past the right edge.
        assert needs_panel_repair([(6728.0, 1321.0), (7015.0, 1502.0)], 4014)

    def test_already_repaired_coords_are_left_alone(self):
        assert not needs_panel_repair([(1761.0, 1321.0), (2048.0, 1502.0)], 4014)

    def test_boundary_x_equal_to_width_counts_as_out_of_bounds(self):
        assert needs_panel_repair([(4014.0, 10.0)], 4014)

    def test_repair_is_idempotent(self):
        pts = [(6728.0, 1321.0), (7015.0, 1502.0)]
        once = repair_panel_offset(pts, 4968.0, 4014)
        twice = repair_panel_offset(once, 4968.0, 4014)
        assert once == twice
        assert all(0 <= x < 4014 for x, _ in once)

    def test_repair_rejects_an_offset_that_lands_out_of_bounds(self):
        # A wrong panel width (e.g. the other slate family's) must fail loudly
        # rather than silently producing negative coordinates.
        with pytest.raises(ValueError, match="out of bounds"):
            repair_panel_offset([(4200.0, 10.0)], 4968.0, 4014)


class TestTemplateCorrespondences:
    """Template point <-> image point pairing, the model's actual target."""

    def test_pairs_visible_template_points_with_label_points(self):
        template = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
        labeled = [(5.0, 5.0), (7.0, 5.0), (9.0, 5.0)]
        pairs = template_correspondences(template, labeled, skipped=[1])
        assert pairs == [
            (0, (0.0, 0.0), (5.0, 5.0)),
            (2, (20.0, 0.0), (7.0, 5.0)),
            (3, (30.0, 0.0), (9.0, 5.0)),
        ]

    def test_length_mismatch_is_rejected(self):
        # A label whose point count doesn't equal len(template) - len(skipped)
        # is unusable for PnP; the extractor must quarantine it rather than
        # silently mis-pair the correspondences.
        with pytest.raises(ValueError, match="expected 3"):
            template_correspondences(
                [(0.0, 0.0)] * 4, [(1.0, 1.0)] * 2, skipped=[0]
            )
