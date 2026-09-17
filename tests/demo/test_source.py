"""Selection: the floor is not negotiable."""

from __future__ import annotations

import pyarrow as pa
from pyiceberg.expressions import And, GreaterThanOrEqual, LessThanOrEqual

from icechest.demo.source import (
    DATETIME_FLOOR,
    _drop_null_proj_transform,
    archive_metadata_url,
    granule_filter,
)


def leaves(expression):
    """Every leaf predicate of a conjunction, in any order."""
    if isinstance(expression, And):
        return leaves(expression.left) + leaves(expression.right)
    return [expression]


def test_metadata_url_is_the_archive_layout():
    assert archive_metadata_url() == (
        "s3://nasa-maap-data-store/file-staging/nasa-map/"
        "hls-stac-geoparquet-archive/v2/HLSL30_2.0/iceberg/metadata/"
        "latest.metadata.json"
    )


def test_filter_always_carries_the_floor_and_the_not_null():
    """proj:shape and proj:transform are null on older records; a granule
    without them would produce a half-declared spatial convention."""
    rendered = str(granule_filter())
    assert DATETIME_FLOOR in rendered
    assert "proj:transform" in rendered
    assert "NotNull" in rendered or "not_null" in rendered.lower()


def test_caller_datetime_narrows_rather_than_replaces():
    rendered = str(
        granule_filter(
            datetime=(
                "2026-03-01T00:00:00+00:00",
                "2026-04-01T00:00:00+00:00",
            )
        )
    )
    assert DATETIME_FLOOR in rendered
    assert "2026-03-01T00:00:00+00:00" in rendered
    assert "2026-04-01T00:00:00+00:00" in rendered


def test_tile_filter_matches_the_granule_id_prefix():
    rendered = str(granule_filter(tile="T20JKP"))
    assert "HLS.L30.T20JKP." in rendered


def test_bbox_filter_pairs_each_bound_with_the_right_field():
    """A transposed comparison would still mention all four fields, so the
    test has to pin which bound each one is compared against."""
    rendered = {
        str(term) for term in leaves(granule_filter(bbox=(-67.0, -29.0, -65.0, -27.0)))
    }

    assert str(GreaterThanOrEqual("bbox.xmax", -67.0)) in rendered
    assert str(LessThanOrEqual("bbox.xmin", -65.0)) in rendered
    assert str(GreaterThanOrEqual("bbox.ymax", -29.0)) in rendered
    assert str(LessThanOrEqual("bbox.ymin", -27.0)) in rendered


def test_require_proj_transform_false_drops_the_not_null_term():
    """select_granules runs the scan this way: PyIceberg's ArrowScan cannot
    project a list-typed column that appears only in the row filter, and
    proj:transform is list<double>. The condition still gets enforced -- just
    client-side, on the returned Arrow table, instead of pushed into the
    scan."""
    rendered = str(granule_filter(require_proj_transform=False))
    assert DATETIME_FLOOR in rendered
    assert "proj:transform" not in rendered


def test_drop_null_proj_transform_keeps_only_populated_rows():
    """This is the other half of the fix above -- the half that actually
    enforces the condition once it is no longer pushed into the scan. A pure
    Arrow table stands in for what the (unfiltered) scan would return: one row
    with a null transform, as an old, pre-2026 record would carry, and one
    with a real one."""
    rows = pa.table(
        {
            "id": ["old-granule", "new-granule"],
            "proj:transform": pa.array(
                [None, [30.0, 0.0, 199980.0, 0.0, -30.0, -3099960.0]],
                type=pa.list_(pa.float64()),
            ),
        }
    )

    kept = _drop_null_proj_transform(rows)

    assert kept["id"].to_pylist() == ["new-granule"]
    assert kept["proj:transform"].to_pylist() == [
        [30.0, 0.0, 199980.0, 0.0, -30.0, -3099960.0]
    ]


def test_caller_range_starting_before_the_floor_still_gets_the_floor():
    """A caller asking for 2020 data must not thereby escape the floor: the
    two conditions are conjoined, not substituted."""
    rendered = {
        str(term)
        for term in leaves(
            granule_filter(
                datetime=(
                    "2020-01-01T00:00:00+00:00",
                    "2020-06-01T00:00:00+00:00",
                )
            )
        )
    }
    assert str(GreaterThanOrEqual("datetime", DATETIME_FLOOR)) in rendered
    assert str(GreaterThanOrEqual("datetime", "2020-01-01T00:00:00+00:00")) in rendered
