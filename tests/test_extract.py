"""Pure-logic tests for the extractor's record shaping + quarantine rules."""

from types import SimpleNamespace

from slate_training.extract import ExtractReport, build_records


def _slate(n_points=6, dpi=300, name="V-Slate 1", slate_id=3):
    return SimpleNamespace(
        id=slate_id,
        name=name,
        dpi=dpi,
        reference_points=[(float(i * 10), 0.0) for i in range(n_points)],
    )


def _label(**kw):
    base = dict(
        id=1,
        image_id=100,
        completed=True,
        superseded=False,
        reference_points=[(float(i), 5.0) for i in range(6)],
        skipped_points=None,
        upside_down=False,
        slate_rectangle=[(0.0, 0.0), (50.0, 50.0)],
        user_id=7,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _ctx(image_ids=(100,), with_laser=True):
    checksums = {i: f"chk{i}" for i in image_ids}
    lasers = {
        i: (SimpleNamespace(x=1.0, y=2.0) if with_laser else None)
        for i in image_ids
    }
    return checksums, lasers


def _run(labels, slate=None, panel=None, photo=None, **kw):
    slate = slate or _slate()
    checksums, lasers = _ctx(**kw)
    report = ExtractReport()
    records = build_records(
        labels, slate, 42, checksums, lasers, 9, report, panel, photo
    )
    return records, report


class TestPanelRepair:
    """The historical corpus is in composite space; the extractor repairs it."""

    def test_composite_points_are_shifted_into_photo_space(self):
        label = _label(
            reference_points=[(4968.0 + i * 10, 5.0) for i in range(6)]
        )
        records, _ = _run([label], panel=4968.0, photo=4014)
        assert records[0].reference_points[0] == (0.0, 5.0)
        assert all(0 <= x < 4014 for x, _ in records[0].reference_points)

    def test_already_repaired_points_pass_through_unchanged(self):
        label = _label(reference_points=[(float(i), 5.0) for i in range(6)])
        records, _ = _run([label], panel=4968.0, photo=4014)
        assert records[0].reference_points[0] == (0.0, 5.0)

    def test_wrong_panel_width_is_quarantined_not_silently_wrong(self):
        label = _label(
            reference_points=[(4100.0 + i, 5.0) for i in range(6)]
        )
        records, report = _run([label], panel=4968.0, photo=4014)
        assert records == []
        assert report.quarantined["panel_repair_out_of_bounds"] == 1


class TestHappyPath:
    def test_keeps_a_well_formed_label(self):
        records, report = _run([_label()])
        assert len(records) == 1
        assert report.records_kept == 1
        assert records[0].checksum == "chk100"
        assert records[0].slate_name == "V-Slate 1"
        assert records[0].dive_id == 42
        assert records[0].camera_id == 9

    def test_records_per_slate_and_per_dive_counts(self):
        records, report = _run([_label()])
        assert report.per_slate_counts == {"V-Slate 1": 1}
        assert report.per_dive_counts == {"42": 1}

    def test_skipped_histogram_buckets_by_count(self):
        # 6 template points, 2 skipped -> label must carry 4 points.
        label = _label(
            skipped_points=[1, 4],
            reference_points=[(float(i), 5.0) for i in range(4)],
        )
        _, report = _run([label])
        assert report.skipped_point_histogram == {"2": 1}


class TestQuarantine:
    def test_incomplete_label_is_dropped(self):
        records, report = _run([_label(completed=False)])
        assert records == []
        assert report.quarantined["not_completed"] == 1
        assert report.labels_completed == 0

    def test_superseded_label_is_dropped(self):
        records, report = _run([_label(superseded=True)])
        assert records == []
        assert report.quarantined["superseded"] == 1

    def test_label_without_points_is_dropped(self):
        records, report = _run([_label(reference_points=None)])
        assert records == []
        assert report.quarantined["no_reference_points"] == 1

    def test_point_count_mismatch_is_quarantined_not_mispaired(self):
        # 6 template points, 1 skipped -> 5 expected, but only 3 given.
        label = _label(
            skipped_points=[0],
            reference_points=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
        )
        records, report = _run([label])
        assert records == []
        assert report.quarantined["point_count_mismatch"] == 1

    def test_missing_checksum_is_dropped(self):
        records, report = _run([_label(image_id=999)])
        assert records == []
        assert report.quarantined["no_image_checksum"] == 1

    def test_missing_laser_is_kept_but_flagged_oracle_only(self):
        # Still a valid keypoint example -- only the calibration oracle is lost.
        records, report = _run([_label()], with_laser=False)
        assert len(records) == 1
        assert records[0].laser_x is None
        assert report.quarantined["no_laser_label_oracle_only"] == 1

    def test_template_without_geometry_is_dropped(self):
        slate = _slate()
        slate.dpi = None
        records, report = _run([_label()], slate=slate)
        assert records == []
        assert report.quarantined["template_missing_geometry"] == 1
