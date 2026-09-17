"""Zarr convention attributes for a virtualized granule.

The arrays carry no georeferencing of their own -- a virtual reference is a
pointer into a COG, not a coordinate system -- so the conventions are what make
the written groups self-describing. Every value here comes from the STAC record,
which is why selection insists on records that actually carry ``proj:transform``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: Each convention's schema fixes every field of its metadata object as a
#: ``const`` and sets ``additionalProperties: false``, so these are reproduced
#: exactly and carry nothing extra. The published tag is ``v0.1``, the names
#: carry no trailing colon -- the colon belongs to the attribute keys, not the
#: convention -- and proj lives under ``zarr-conventions``, not the
#: ``zarr-experimental/geo-proj`` location that now only redirects.
MULTISCALES_CONVENTION: dict[str, str] = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md",
    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
    "name": "multiscales",
    "description": "Multiscale layout of zarr datasets",
}

PROJ_CONVENTION: dict[str, str] = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/proj/blob/v0.1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj",
    "description": "Coordinate reference system information for geospatial data",
}

SPATIAL_CONVENTION: dict[str, str] = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v0.1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial",
    "description": "Spatial coordinate information",
}


#: The group each band's resolution levels are written into, as a child of the
#: band group. One constant governs both the path written and the paths
#: declared, because a layout entry naming a path nothing was written to is a
#: pyramid declaration that points at nothing.
MULTISCALES_GROUP = "multiscales"

#: ``asset`` and ``derived_from`` are relative to the group carrying these
#: attributes, which is the band group -- two above the arrays themselves,
#: since each level is a group of its own.
LEVEL_ASSET_PREFIX = f"{MULTISCALES_GROUP}/"


def multiscale_layout(
    levels: int,
    *,
    array_name: str,
    prefix: str = LEVEL_ASSET_PREFIX,
) -> list[dict[str, Any]]:
    """Describe the COG's overview pyramid, one entry per resolution level.

    Each level is a group holding one array, and an ``asset`` resolves to the
    array rather than to the group above it -- the convention's nested-array
    layout, ``{"asset": "0/data"}``. Levels could be sibling arrays instead;
    the convention permits that too, and calls it the natural translation of
    COG overviews. They are groups because levels differ in y and x while
    dimensions of one name must agree within a node, so sibling arrays cannot
    be opened together as a dataset or a DataTree at all.

    A layout entry's ``transform`` is defined *relative to* ``derived_from``,
    so each overview's step from the level above it is a factor of two in both
    axes -- not the level's absolute affine, which is not a valid layout
    transform at all. A per-level absolute affine has a home if one is ever
    wanted: the spatial convention permits ``spatial:shape`` and
    ``spatial:transform`` overrides inside a layout item.

    ``prefix`` is prepended to every path, because the paths are resolved
    relative to whichever group these attributes are written to.
    """
    layout: list[dict[str, Any]] = [{"asset": f"{prefix}0/{array_name}"}]
    for level in range(1, levels):
        layout.append(
            {
                "asset": f"{prefix}{level}/{array_name}",
                "derived_from": f"{prefix}{level - 1}/{array_name}",
                "transform": {"scale": [2, 2]},
            }
        )
    return layout


def granule_attrs(
    *,
    epsg: int,
    shape: Sequence[int],
    transform: Sequence[float],
    levels: int,
    array_name: str,
    prefix: str = LEVEL_ASSET_PREFIX,
) -> dict[str, Any]:
    """Build the convention attributes for one band group.

    ``array_name`` is what the array inside each level group is called, which
    is what an ``asset`` path has to end in.

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
        # The pixel counts are the *shape*; "dimensions" is the pair of
        # dimension names, in row-major order.
        "spatial:shape": [rows, cols],
        "spatial:dimensions": ["y", "x"],
        "spatial:transform": affine,
        "spatial:bbox": [xmin, ymin, xmax, ymax],
        "multiscales": {
            "layout": multiscale_layout(levels, array_name=array_name, prefix=prefix)
        },
    }
