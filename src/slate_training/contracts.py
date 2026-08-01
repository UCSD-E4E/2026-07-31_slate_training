"""The `DiveSlateLabel` coordinate and ordering contract.

Everything downstream — the extractor, the classical baseline, the model's
target encoding, and the Label Studio pre-annotation seeding — depends on
getting these three facts right, so they live in one small, tested module
rather than being re-derived at each call site.

Verified against fishsense-lite @ 2026-07-31:

**Coordinate space.** The slate labeling task shows a *composite* canvas:
the slate template PDF rendered on the left, the rectified photo on the
right (`preprocess_slate_image.composite_slate_with_image`). The labeler
clicks on the photo half, so raw LS coordinates carry a horizontal offset
of the rendered panel width.

`sync_dive_slate_labels_...` is *supposed* to subtract that offset before
persisting — but **all 104 completed production labels were stored without
it** (measured from the 2026-07-31 dump; verified visually in
`scripts/verify_label_overlay.py`). So the corpus as it exists is in
*composite* pixels, and every consumer must repair it via
`repair_panel_offset` before doing geometry. `LaserLabel.x/y`, by contrast,
really is in photo pixels — so the two are currently in different spaces.

The offset is exactly recoverable per template
(`composite_panel_width`), and differs by template family: 4968 px for the
V-Slates, 3903 px for Tic-Tac-Toe 6, against a 4014×3016 photo.

**Ordering.** `reference_points` is a dense list positionally aligned with
`DiveSlate.reference_points` once `skipped_points` are removed. Stage 13
builds its solvePnP correspondences purely by position, so a point count
that disagrees with ``len(template) - len(skipped)`` silently mis-pairs
geometry. `template_correspondences` makes that a hard error instead.

**`slate_rectangle` is a 2-point axis-aligned bbox** ``[(x0,y0),(x1,y1)]``
derived from an LS `RectangleLabels` control — not a 4-corner quad. There
is therefore no labeled quad to fit a homography to; the pose prior has to
come from the reference points themselves.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]

__all__ = [
    "Point",
    "composite_panel_width",
    "photo_to_composite",
    "composite_to_photo",
    "needs_panel_repair",
    "repair_panel_offset",
    "drop_skipped",
    "template_correspondences",
]


def composite_panel_width(pdf_aspect_ratio: float, photo_height: float) -> float:
    """Pixel width of the template panel inside the LS composite canvas.

    The compositor scales the render by ``photo_height / pdf_height``, so the
    panel width is ``(pdf_width / pdf_height) * photo_height``. DPI cancels
    out, which is why the aspect ratio alone is enough — mirrors
    `compute_pdf_panel_width_in_composite` in the sync activity.
    """
    return float(pdf_aspect_ratio) * float(photo_height)


def photo_to_composite(points: Iterable[Point], panel_width: float) -> List[Point]:
    """Shift rectified-photo points right into composite-canvas space.

    Use when seeding a prediction as an LS pre-annotation: the stored/predicted
    space is the photo, but LS renders the composite.
    """
    return [(float(x) + float(panel_width), float(y)) for x, y in points]


def composite_to_photo(points: Iterable[Point], panel_width: float) -> List[Point]:
    """Shift composite-canvas points left into rectified-photo space.

    The direction the sync activity applies when persisting a human label.
    """
    return [(float(x) - float(panel_width), float(y)) for x, y in points]


def needs_panel_repair(points: Sequence[Point], photo_width: float) -> bool:
    """True when `points` still carry the composite panel offset.

    The tell is unambiguous — a point on the photo cannot have ``x >=
    photo_width``, but every composite-space point does, because the panel is
    wider than the photo for all current templates. Used to make
    `repair_panel_offset` idempotent and to guard against double-shifting a
    corpus that gets fixed upstream later.
    """
    return any(float(x) >= float(photo_width) for x, _ in points)


def repair_panel_offset(
    points: Sequence[Point], panel_width: float, photo_width: float
) -> List[Point]:
    """Shift composite-space points into photo space, idempotently.

    Returns `points` unchanged when they are already in photo space. Raises
    when the shift would land outside the frame — that means the wrong
    template's panel width was used, which must fail loudly rather than
    silently corrupt the geometry.
    """
    if not needs_panel_repair(points, photo_width):
        return [(float(x), float(y)) for x, y in points]

    shifted = composite_to_photo(points, panel_width)
    if any(not 0 <= x < float(photo_width) for x, _ in shifted):
        raise ValueError(
            f"panel_width={panel_width} puts points out of bounds for a "
            f"{photo_width}px-wide photo; wrong template?"
        )
    return shifted


def drop_skipped(points: Sequence[Point], skipped: Iterable[int]) -> List[Point]:
    """Remove `skipped` indices from `points`, interpreting them against the
    *original* list.

    Note this deliberately differs from stage 13, which does a sequential
    ``list.pop(idx)``. That is only correct for zero or one skipped point:
    with ``skipped=[2, 5]`` the pop of 2 slides every later element down one,
    so the pop of 5 removes what was originally index 6. Building a training
    target or an eval oracle on that would bake in a mis-pairing, so this
    resolves all indices up front. See `docs/design.md` for the upstream
    bug report.
    """
    indices = [int(i) for i in skipped]

    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate skipped index in {indices}")
    for i in indices:
        if not 0 <= i < len(points):
            raise ValueError(
                f"skipped index {i} out of range for {len(points)} points"
            )

    drop = set(indices)
    return [p for i, p in enumerate(points) if i not in drop]


def template_correspondences(
    template_points: Sequence[Point],
    labeled_points: Sequence[Point],
    skipped: Iterable[int],
) -> List[Tuple[int, Point, Point]]:
    """Pair each visible template point with its labeled image point.

    Returns ``(template_index, template_point, image_point)`` triples — the
    template index is retained so a prediction can be scattered back into a
    full-length, skip-aware label.

    Raises `ValueError` when the labeled point count disagrees with the
    template minus skips, which is the one failure mode that would otherwise
    corrupt geometry silently.
    """
    visible = [
        (i, p)
        for i, p in enumerate(template_points)
        if i not in set(int(s) for s in skipped)
    ]
    # Re-validate indices through drop_skipped for its range/duplicate checks.
    drop_skipped(template_points, skipped)

    if len(labeled_points) != len(visible):
        raise ValueError(
            f"expected {len(visible)} labeled points "
            f"({len(template_points)} template - {len(set(int(s) for s in skipped))} skipped), "
            f"got {len(labeled_points)}"
        )

    return [
        (idx, tpl, (float(x), float(y)))
        for (idx, tpl), (x, y) in zip(visible, labeled_points)
    ]
