"""Zarr convention attributes for a virtualized granule.

The arrays carry no georeferencing of their own -- a virtual reference is a
pointer into a COG, not a coordinate system -- so the conventions are what make
the written groups self-describing. Every value here comes from the STAC record,
which is why selection insists on records that actually carry ``proj:transform``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

MULTISCALES_CONVENTION: dict[str, str] = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
    "name": "multiscales",
    "description": "Multiscale layout of zarr datasets",
}

PROJ_CONVENTION: dict[str, str] = {
    "schema_url": "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj:",
    "description": "Coordinate reference system information for geospatial data",
}

SPATIAL_CONVENTION: dict[str, str] = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial:",
    "description": "Spatial coordinate information",
}


def multiscale_layout(
    transform: Sequence[float], levels: int
) -> list[dict[str, Any]]:
    """Describe the COG's overview pyramid, one entry per resolution level.

    Each overview halves both dimensions, so level *n* has pixels 2**n times
    larger than level 0 while sharing its origin.
    """
    layout: list[dict[str, Any]] = [{"asset": "0"}]
    for level in range(1, levels):
        factor = 2**level
        layout.append(
            {
                "asset": str(level),
                "derived_from": str(level - 1),
                "factors": [2, 2],
                "transform": [
                    transform[0] * factor,
                    transform[1],
                    transform[2],
                    transform[3],
                    transform[4] * factor,
                    transform[5],
                ],
            }
        )
    return layout


def granule_attrs(
    *,
    epsg: int,
    shape: Sequence[int],
    transform: Sequence[float],
    levels: int,
) -> dict[str, Any]:
    """Build the convention attributes for one band group.

    ``transform`` is the archive's nine-element row-major affine; the conventions
    want the six that carry the affine itself. The projected bounds fall out of
    the transform and shape exactly, so the record's geographic ``bbox`` -- which
    would need reprojecting -- is not used.
    """
    rows, cols = int(shape[0]), int(shape[1])
    affine = [float(value) for value in transform[:6]]
    xmin, ymax = affine[2], affine[5]
    xmax = xmin + cols * affine[0]
    ymin = ymax + rows * affine[4]  # affine[4] is negative: north-up
    return {
        "zarr_conventions": [
            MULTISCALES_CONVENTION,
            PROJ_CONVENTION,
            SPATIAL_CONVENTION,
        ],
        "proj:code": f"EPSG:{int(epsg)}",
        "spatial:dimensions": [rows, cols],
        "spatial:transform": affine,
        "spatial:bbox": [xmin, ymin, xmax, ymax],
        "multiscales": {"layout": multiscale_layout(affine, levels)},
    }
