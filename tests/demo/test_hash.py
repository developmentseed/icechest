"""The spatio-temporal hash column, computed per record from its own fields."""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa
import pytest

from icechest.demo.hash import (
    HASH_END,
    HASH_START,
    PARTITION_BITS,
    hasher,
    with_stac_hash,
)


def rows(*specs):
    """Build granule rows from (id, datetime, xmin, ymin, xmax, ymax) tuples."""
    return pa.table(
        {
            "id": [s[0] for s in specs],
            "datetime": pa.array([s[1] for s in specs], pa.timestamp("us", tz="UTC")),
            "bbox": [
                {"xmin": s[2], "ymin": s[3], "xmax": s[4], "ymax": s[5]} for s in specs
            ],
        }
    )


def test_adds_a_long_column_named_stac_hash():
    table = with_stac_hash(
        rows(("a", datetime(2026, 1, 4, tzinfo=UTC), -66.0, -29.0, -65.0, -28.0))
    )

    assert "stac_hash" in table.column_names
    assert table.schema.field("stac_hash").type == pa.int64()


def test_hash_is_taken_at_the_bbox_centre():
    """The point is the centre of the footprint, not one of its corners --
    a corner would put a granule's hash in a neighbouring cell."""
    table = with_stac_hash(
        rows(("a", datetime(2026, 1, 4, tzinfo=UTC), -66.0, -29.0, -64.0, -27.0))
    )

    expected = hasher().hash(datetime(2026, 1, 4, tzinfo=UTC), -65.0, -28.0)
    assert table["stac_hash"][0].as_py() == expected


def test_near_neighbours_hash_closer_than_distant_granules():
    """The point of a Morton code: adjacency in space and time survives into
    the integer, which is what makes it worth sorting on."""
    when = datetime(2026, 1, 4, tzinfo=UTC)
    table = with_stac_hash(
        rows(
            ("here", when, -66.0, -29.0, -65.0, -28.0),
            ("next-door", when, -65.9, -28.9, -64.9, -27.9),
            ("far", when, 100.0, 40.0, 101.0, 41.0),
        )
    )
    here, next_door, far = (v.as_py() for v in table["stac_hash"])

    assert abs(next_door - here) < abs(far - here)


def test_every_row_gets_its_own_hash():
    when = datetime(2026, 1, 4, tzinfo=UTC)
    table = with_stac_hash(
        rows(
            ("a", when, -66.0, -29.0, -65.0, -28.0),
            ("b", when, 10.0, 10.0, 11.0, 11.0),
            ("c", when, 100.0, 40.0, 101.0, 41.0),
        )
    )

    assert table.num_rows == 3
    assert len({v.as_py() for v in table["stac_hash"]}) == 3


def test_out_of_extent_datetimes_are_clamped_rather_than_fatal():
    """One stray granule outside the extent must not fail a whole batch, so
    the hash clamps onto the boundary instead of raising."""
    before = datetime(1999, 1, 1, tzinfo=UTC)
    table = with_stac_hash(rows(("ancient", before, -66.0, -29.0, -65.0, -28.0)))

    at_start = hasher().hash(HASH_START, -65.5, -28.5)
    assert table["stac_hash"][0].as_py() == at_start


def test_extent_is_the_whole_globe_over_the_mission_era():
    assert HASH_START == datetime(2013, 1, 1, tzinfo=UTC)
    assert HASH_END == datetime(2036, 1, 1, tzinfo=UTC)
    # A point at each corner of the world hashes without raising, which is what
    # "whole globe" has to mean for a hasher that rejects out-of-extent input.
    corners = hasher()
    for lon, lat in ((-180.0, -90.0), (180.0, 90.0)):
        corners.hash_clamped(datetime(2026, 1, 1, tzinfo=UTC), lon, lat)


def test_hashes_fit_in_a_signed_64_bit_column():
    """Iceberg has no unsigned integer; a hash above int64 max would wrap
    negative and invert the sort order it exists to provide."""
    table = with_stac_hash(
        rows(
            ("low", HASH_START, -180.0, -90.0, -180.0, -90.0),
            ("high", datetime(2035, 12, 31, tzinfo=UTC), 179.9, 89.9, 180.0, 90.0),
        )
    )

    for value in table["stac_hash"]:
        assert 0 <= value.as_py() <= 2**63 - 1


def test_rejects_rows_without_a_bbox():
    """A row with no footprint has no point to hash, and silently hashing it
    at the origin would put it in a cell it has nothing to do with."""
    table = pa.table(
        {
            "id": ["a"],
            "datetime": pa.array(
                [datetime(2026, 1, 4, tzinfo=UTC)], pa.timestamp("us", tz="UTC")
            ),
            "bbox": pa.array([None], pa.struct([("xmin", pa.float64())])),
        }
    )

    with pytest.raises(ValueError, match="bbox"):
        with_stac_hash(table)


def test_block_is_the_high_order_bits_of_the_hash():
    """The partition key. Iceberg's truncate transform takes a 32-bit width, so
    it cannot express a block of a 63-bit Morton code -- the block is computed
    here instead and partitioned by identity."""
    table = with_stac_hash(
        rows(("a", datetime(2026, 1, 4, tzinfo=UTC), -66.0, -29.0, -65.0, -28.0))
    )

    value = table["stac_hash"][0].as_py()
    assert table["stac_hash_block"][0].as_py() == value >> (63 - PARTITION_BITS)


def test_near_neighbours_share_a_block_and_distant_ones_do_not():
    when = datetime(2026, 1, 4, tzinfo=UTC)
    table = with_stac_hash(
        rows(
            ("here", when, -66.0, -29.0, -65.0, -28.0),
            ("next-door", when, -65.9, -28.9, -64.9, -27.9),
            ("far", when, 100.0, 40.0, 101.0, 41.0),
        )
    )
    here, next_door, far = (v.as_py() for v in table["stac_hash_block"])

    assert next_door == here
    assert far != here
