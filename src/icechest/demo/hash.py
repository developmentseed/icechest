"""The spatio-temporal hash each granule row carries.

A Morton code over (datetime, point) that keeps spatially and temporally
adjacent granules adjacent as integers, which is what makes it worth sorting
and partitioning on. It is a clustering key and not an identity: the extent is
divided into 2**21 steps per dimension, so granules within roughly six minutes
of each other at the same point share a hash.

The extent is fixed here rather than derived from the data, because it is what
the hash means. Two rows hashed against different extents are on different
scales, so changing it silently invalidates every hash already written along
with the sort order and partition boundaries built on them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

import pyarrow as pa
import pyarrow.compute as pc
from stac_hash import Hasher

#: The HLS mission era with headroom, over the whole globe. Gives roughly 19 m
#: spatial and six minute temporal resolution.
HASH_START = datetime(2013, 1, 1, tzinfo=UTC)
HASH_END = datetime(2036, 1, 1, tzinfo=UTC)

HASH_COLUMN = "stac_hash"
HASH_BLOCK_COLUMN = "stac_hash_block"

#: A block is the top twelve bits of the hash: four levels of the Morton
#: octree, so sixteen steps per dimension -- roughly 22 degrees of longitude,
#: 11 of latitude and seventeen months. Coarse enough that a batch lands in a
#: handful of partitions rather than one per granule.
#:
#: The block is a column of its own rather than an Iceberg ``truncate``
#: transform because that transform's width is a 32-bit parameter: the widest
#: block it can express of a 63-bit code still leaves billions of partitions.
PARTITION_BITS = 12


@lru_cache(maxsize=1)
def hasher() -> Hasher:
    """The hasher every row in this store is hashed against."""
    return Hasher(HASH_START, HASH_END)


def with_stac_hash(rows: pa.Table) -> pa.Table:
    """Return ``rows`` with its ``stac_hash`` and ``stac_hash_block`` columns.

    Both are computed per record. The point is the centre of each granule's own
    bbox: a corner would place the hash in a cell the granule only touches.
    Out-of-extent datetimes clamp onto the boundary rather than raising, so one
    stray granule cannot fail a batch that is otherwise fine.
    """
    bbox = rows["bbox"]
    if bbox.null_count:
        raise ValueError(
            f"{bbox.null_count} row(s) have no bbox, so there is no point to "
            "hash; a granule without a footprint cannot be placed in space"
        )

    def centre(low: str, high: str) -> list[float]:
        return pc.divide(
            pc.add(pc.struct_field(bbox, low), pc.struct_field(bbox, high)),
            2.0,
        ).to_pylist()

    hashes = hasher().hash_all_clamped(
        rows["datetime"].to_pylist(),
        centre("xmin", "xmax"),
        centre("ymin", "ymax"),
    )
    shift = 63 - PARTITION_BITS
    return rows.append_column(HASH_COLUMN, pa.array(hashes, pa.int64())).append_column(
        HASH_BLOCK_COLUMN, pa.array([value >> shift for value in hashes], pa.int64())
    )
