"""Selection: the floor is not negotiable."""

from __future__ import annotations

from icechest.demo.source import (
    DATETIME_FLOOR,
    archive_metadata_url,
    granule_filter,
)


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


def test_bbox_filter_uses_all_four_bounds():
    rendered = str(granule_filter(bbox=(-67.0, -29.0, -65.0, -27.0)))
    for bound in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax"):
        assert bound in rendered
