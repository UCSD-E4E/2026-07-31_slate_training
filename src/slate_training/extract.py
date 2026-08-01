"""Pull the `DiveSlateLabel` dataset + template geometry out of the FishSense API.

Deliberately split into a thin async I/O shell (`extract`) and a pure
`build_records` core so the record-shaping logic — which is where the
contract violations show up — is testable without credentials or network.

Writes a single vendored JSON manifest (`data/slate_dataset.json`) following
the `2026-07-18_fishsense-core-test` precedent: labels are small and vendored,
images stay out of the repo and are fetched separately by checksum.

Usage:
    uv run python -m slate_training.extract --out data/slate_dataset.json

Credentials come from the same place the workers read them (dynaconf
`settings.fishsense_api.{url,username,password}`) or the environment:
    FISHSENSE_API_URL, FISHSENSE_API_USERNAME, FISHSENSE_API_PASSWORD
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

from slate_training.contracts import (
    Point,
    repair_panel_offset,
    template_correspondences,
)

# Garage prefix the data-worker writes slate composites to (stage 9). The raw
# .ORF for the same frame is addressed by the same checksum.
DIVE_SLATE_FOLDER = "preprocess_slate_images_jpeg"


@dataclass
class SlateRecord:
    """One usable (image, slate, label) training/eval example."""

    image_id: int
    dive_id: int
    checksum: str
    slate_id: int
    slate_name: str
    slate_dpi: int
    # Rectified-photo pixels, dense, aligned to the visible template points.
    reference_points: List[Point]
    skipped_points: List[int]
    upside_down: bool | None
    slate_rectangle: List[Point] | None
    # Oracle wiring: the laser dot on this frame + the human calibration the
    # dive already has. Together these let us score a predicted labeling by
    # the calibration it produces rather than by pixel error.
    laser_x: float | None
    laser_y: float | None
    camera_id: int | None
    label_id: int | None = None
    user_id: int | None = None


@dataclass
class ExtractReport:
    """Counts + quarantine reasons; the data report in deliverable #1."""

    dives_scanned: int = 0
    labels_seen: int = 0
    labels_completed: int = 0
    records_kept: int = 0
    quarantined: Dict[str, int] = field(default_factory=dict)
    per_slate_counts: Dict[str, int] = field(default_factory=dict)
    per_dive_counts: Dict[str, int] = field(default_factory=dict)
    skipped_point_histogram: Dict[str, int] = field(default_factory=dict)
    dives_with_human_calibration: int = 0

    def quarantine(self, reason: str) -> None:
        self.quarantined[reason] = self.quarantined.get(reason, 0) + 1


def build_records(
    labels: Sequence[Any],
    slate: Any,
    dive_id: int,
    checksum_by_image_id: Dict[int, str],
    laser_by_image_id: Dict[int, Any],
    camera_id: int | None,
    report: ExtractReport,
    panel_width: float | None = None,
    photo_width: float | None = None,
) -> List[SlateRecord]:
    """Shape one dive's labels into records, quarantining unusable ones.

    Pure: no I/O. Every rejection is counted in `report.quarantined` so the
    data report explains its own coverage rather than silently dropping rows.

    When `panel_width` and `photo_width` are given, points are repaired out of
    composite space into photo space. The entire historical corpus needs this
    — see `contracts.repair_panel_offset`. The repair is idempotent, so
    passing them is safe even once the upstream sync is fixed.
    """
    template = list(slate.reference_points or [])
    records: List[SlateRecord] = []

    for label in labels:
        report.labels_seen += 1

        if not label.completed:
            report.quarantine("not_completed")
            continue
        report.labels_completed += 1

        if label.superseded:
            report.quarantine("superseded")
            continue
        if not label.reference_points:
            report.quarantine("no_reference_points")
            continue
        if label.image_id is None:
            report.quarantine("no_image_id")
            continue
        if not template or not slate.dpi:
            report.quarantine("template_missing_geometry")
            continue

        skipped = list(label.skipped_points or [])

        points = [tuple(p) for p in label.reference_points]
        if panel_width is not None and photo_width is not None:
            try:
                points = repair_panel_offset(points, panel_width, photo_width)
            except ValueError:
                report.quarantine("panel_repair_out_of_bounds")
                continue

        # The contract check. A count mismatch means the positional pairing
        # stage 13 relies on is broken for this row -- unusable as either a
        # training target or an eval oracle.
        try:
            template_correspondences(template, points, skipped)
        except ValueError:
            report.quarantine("point_count_mismatch")
            continue

        checksum = checksum_by_image_id.get(label.image_id)
        if checksum is None:
            report.quarantine("no_image_checksum")
            continue

        laser = laser_by_image_id.get(label.image_id)
        laser_x = getattr(laser, "x", None) if laser is not None else None
        laser_y = getattr(laser, "y", None) if laser is not None else None
        if laser_x is None or laser_y is None:
            # Still a valid keypoint example, just not usable for the
            # end-to-end calibration oracle. Counted, not dropped.
            report.quarantine("no_laser_label_oracle_only")

        records.append(
            SlateRecord(
                image_id=label.image_id,
                dive_id=dive_id,
                checksum=checksum,
                slate_id=slate.id,
                slate_name=slate.name,
                slate_dpi=int(slate.dpi),
                reference_points=points,
                skipped_points=skipped,
                upside_down=label.upside_down,
                slate_rectangle=(
                    [tuple(p) for p in label.slate_rectangle]
                    if label.slate_rectangle
                    else None
                ),
                laser_x=laser_x,
                laser_y=laser_y,
                camera_id=camera_id,
                label_id=label.id,
                user_id=label.user_id,
            )
        )

        key = slate.name
        report.per_slate_counts[key] = report.per_slate_counts.get(key, 0) + 1
        dkey = str(dive_id)
        report.per_dive_counts[dkey] = report.per_dive_counts.get(dkey, 0) + 1
        skey = str(len(skipped))
        report.skipped_point_histogram[skey] = (
            report.skipped_point_histogram.get(skey, 0) + 1
        )

    report.records_kept += len(records)
    return records


def _client():
    """Build an SDK client from dynaconf settings, falling back to env vars."""
    from fishsense_api_sdk.client import Client  # noqa: PLC0415

    url = os.environ.get("FISHSENSE_API_URL")
    user = os.environ.get("FISHSENSE_API_USERNAME")
    password = os.environ.get("FISHSENSE_API_PASSWORD")

    if not (url and user and password):
        raise SystemExit(
            "Set FISHSENSE_API_URL / _USERNAME / _PASSWORD (or wire dynaconf) "
            "before running the extractor."
        )
    return Client(url, user, password)


async def extract(out_path: str) -> ExtractReport:
    """Walk every dive with a slate, collect its labels, and write the manifest."""
    report = ExtractReport()
    records: List[SlateRecord] = []

    async with _client() as fs:
        slates = await fs.dive_slates.get() or []
        slate_by_id = {s.id: s for s in slates}

        dives = await fs.dives.get() or []
        if not isinstance(dives, list):
            dives = [dives]

        for dive in dives:
            if getattr(dive, "dive_slate_id", None) is None:
                continue
            slate = slate_by_id.get(dive.dive_slate_id)
            if slate is None:
                report.quarantine("dive_slate_id_not_found")
                continue

            report.dives_scanned += 1

            labels = await fs.labels.get_dive_slate_labels(dive.id) or []

            checksum_by_image_id: Dict[int, str] = {}
            laser_by_image_id: Dict[int, Any] = {}
            for label in labels:
                if label.image_id is None:
                    continue
                image = await fs.images.get(image_id=label.image_id)
                if image is not None and image.checksum:
                    checksum_by_image_id[label.image_id] = image.checksum
                laser_by_image_id[label.image_id] = await fs.labels.get_laser_label(
                    image_id=label.image_id
                )

            # The end-to-end oracle: the extrinsics a human labeling produced.
            extrinsics = await fs.dives.get_laser_extrinsics(dive.id)
            if extrinsics is not None:
                report.dives_with_human_calibration += 1

            records.extend(
                build_records(
                    labels,
                    slate,
                    dive.id,
                    checksum_by_image_id,
                    laser_by_image_id,
                    getattr(dive, "camera_id", None),
                    report,
                )
            )

        templates = [
            {
                "id": s.id,
                "name": s.name,
                "dpi": s.dpi,
                "path": s.path,
                "reference_points": [tuple(p) for p in (s.reference_points or [])],
                "n_reference_points": len(s.reference_points or []),
            }
            for s in slates
        ]

    payload = {
        "folder": DIVE_SLATE_FOLDER,
        "templates": templates,
        "records": [asdict(r) for r in records],
        "report": asdict(report),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/slate_dataset.json")
    args = parser.parse_args()

    report = asyncio.run(extract(args.out))
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
