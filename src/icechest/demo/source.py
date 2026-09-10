"""Reading granule records from the MAAP HLS STAC-geoparquet archive.

The archive publishes a static Iceberg ``metadata.json``, so PyIceberg reads it
with no catalog service in the path -- the same shape icechest argues for, which
is why it needs no DuckDB or bespoke parquet globbing here.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from pyiceberg.expressions import (
    And,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    NotNull,
    StartsWith,
)
from pyiceberg.table import StaticTable

if TYPE_CHECKING:
    import pyarrow as pa
    from pyiceberg.expressions import BooleanExpression

ARCHIVE_ROOT = (
    "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive"
)

#: Records older than this carry a null ``proj:transform`` and ``proj:shape``,
#: so their spatial convention could not be populated. Verified against the
#: archive: null in a 2000-row sample of 2013, populated in 3000 rows of 2026.
DATETIME_FLOOR = "2026-01-01T00:00:00+00:00"


def archive_metadata_url(
    collection: str = "HLSL30_2.0", version: str = "v2"
) -> str:
    return (
        f"{ARCHIVE_ROOT}/{version}/{collection}/iceberg/metadata/latest.metadata.json"
    )


def open_archive(
    collection: str = "HLSL30_2.0",
    version: str = "v2",
    region: str = "us-west-2",
) -> StaticTable:
    """Open the archive table without a catalog."""
    return StaticTable.from_metadata(
        archive_metadata_url(collection, version), properties={"s3.region": region}
    )


def granule_filter(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    tile: str | None = None,
    datetime: tuple[str, str] | None = None,
) -> BooleanExpression:
    """Build the selection predicate.

    The date floor and the not-null condition are always present: a caller's
    ``datetime`` range is intersected with the floor, never substituted for it.
    The tile prefix is HLSL30-specific, matching this demo's collection.
    """
    terms: list[BooleanExpression] = [
        GreaterThanOrEqual("datetime", DATETIME_FLOOR),
        NotNull("proj:transform"),
    ]
    if datetime is not None:
        start, end = datetime
        terms.append(GreaterThanOrEqual("datetime", start))
        terms.append(LessThan("datetime", end))
    if bbox is not None:
        west, south, east, north = bbox
        terms.append(GreaterThanOrEqual("bbox.xmax", west))
        terms.append(LessThanOrEqual("bbox.xmin", east))
        terms.append(GreaterThanOrEqual("bbox.ymax", south))
        terms.append(LessThanOrEqual("bbox.ymin", north))
    if tile is not None:
        terms.append(StartsWith("id", f"HLS.L30.{tile}."))
    return functools.reduce(And, terms)


def select_granules(
    table: StaticTable,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    tile: str | None = None,
    datetime: tuple[str, str] | None = None,
    limit: int = 75,
) -> pa.Table:
    """Return granule records in the archive's own schema."""
    scan = table.scan(
        row_filter=granule_filter(bbox=bbox, tile=tile, datetime=datetime),
        limit=limit,
    )
    return scan.to_arrow()
