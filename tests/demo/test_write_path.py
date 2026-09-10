"""The real write path, driven offline.

``write_granule`` is what actually decides where an array lands and where its
pyramid declaration says it landed. Everything else in the batch path was
covered by a stub writer, which is how the arrays came to be written one group
deeper than the layout declared them without a test noticing.

No network is involved: a ``ManifestArray`` over a ``ChunkManifest`` is just a
recorded byte range, and the container the URL falls under is declared when the
store is opened. Nothing reads the referenced object.
"""

from __future__ import annotations

import functools
from datetime import UTC, datetime

import pyarrow as pa
import pytest
import xarray as xr
import zarr
from pyiceberg.schema import Schema
from pyiceberg.types import (
    DoubleType,
    IntegerType,
    ListType,
    NestedField,
    StringType,
    StructType,
    TimestamptzType,
)
from virtualizarr.manifests import ChunkManifest, ManifestArray
from zarr.codecs import BytesCodec
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.dtype import parse_data_type

from icechest.demo.conventions import MULTISCALES_GROUP
from icechest.demo.ingest import ingest_batch
from icechest.demo.store import ensure_table, open_store
from icechest.demo.virtualize import write_granule
from tests.demo.test_virtualize import HREF, build_tiff

GRANULE = "HLS.L30.T20JKP.2026004T142004.v2.0"
S3_URL = (
    "s3://lp-prod-protected/HLSL30.020/"
    "HLS.L30.T20JKP.2026004T142004.v2.0/"
    "HLS.L30.T20JKP.2026004T142004.v2.0.B04.tif"
)
#: A three-level pyramid, deliberately not square so a transposition shows.
SIZES = [(800, 500), (400, 250), (200, 125)]

#: The archive's shape, cut down to the fields the writer actually reads.
SOURCE = Schema(
    NestedField(field_id=1, name="id", field_type=StringType(), required=False),
    NestedField(field_id=2, name="proj:epsg", field_type=IntegerType(), required=False),
    NestedField(
        field_id=3,
        name="proj:shape",
        field_type=ListType(
            element_id=4, element_type=IntegerType(), element_required=False
        ),
        required=False,
    ),
    NestedField(
        field_id=5,
        name="proj:transform",
        field_type=ListType(
            element_id=6, element_type=DoubleType(), element_required=False
        ),
        required=False,
    ),
    NestedField(
        field_id=20, name="datetime", field_type=TimestamptzType(), required=False
    ),
    NestedField(
        field_id=21,
        name="bbox",
        field_type=StructType(
            NestedField(field_id=22, name="xmin", field_type=DoubleType()),
            NestedField(field_id=23, name="ymin", field_type=DoubleType()),
            NestedField(field_id=24, name="xmax", field_type=DoubleType()),
            NestedField(field_id=25, name="ymax", field_type=DoubleType()),
        ),
        required=False,
    ),
    NestedField(
        field_id=7,
        name="assets",
        field_type=StructType(
            NestedField(
                field_id=8,
                name="B04",
                field_type=StructType(
                    NestedField(
                        field_id=9,
                        name="href",
                        field_type=StringType(),
                        required=False,
                    )
                ),
                required=False,
            )
        ),
        required=False,
    ),
)


def virtual_level(level: int) -> xr.Dataset:
    """What ``VirtualTIFF(ifd=level)`` produces: one variable named ``str(level)``."""
    width, height = SIZES[level]
    metadata = ArrayV3Metadata(
        shape=(height, width),
        data_type=parse_data_type("uint16", zarr_format=3),
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": (height, width)},
        },
        chunk_key_encoding={"name": "default"},
        fill_value=0,
        codecs=[BytesCodec()],
        attributes={},
        dimension_names=("y", "x"),
    )
    array = ManifestArray(
        metadata=metadata,
        chunkmanifest=ChunkManifest(
            {"0.0": {"path": S3_URL, "offset": 0, "length": 1024}}
        ),
    )
    return xr.Dataset({str(level): xr.Variable(("y", "x"), array)})


def row(bands=("B04", "B03")):
    return {
        "id": GRANULE,
        "proj:epsg": 32620,
        "proj:shape": [500, 800],
        "proj:transform": [30.0, 0.0, 199980.0, 0.0, -30.0, -3099960.0, 0.0, 0.0, 1.0],
        # datetime and bbox are what the stac_hash column is computed from.
        "datetime": datetime(2026, 1, 4, tzinfo=UTC),
        "bbox": {"xmin": -66.1, "ymin": -29.0, "xmax": -65.3, "ymax": -28.0},
        "assets": {band: {"href": HREF} for band in bands},
    }


def seams():
    """The two network-touching calls, answered from the synthetic pyramid."""
    return {
        "opener": lambda url, registry, ifd: virtual_level(ifd),
        "header_reader": lambda url, registry: build_tiff(SIZES),
    }


@pytest.fixture
def written(tmp_path):
    """One granule, written by the real writer, read back from the commit."""
    repo = open_store(tmp_path)
    tx = repo.transaction("main", "write one granule")
    write_granule(tx, row(), registry=None, bands=("B04", "B03"), **seams())
    tx.session.commit("write one granule")
    return repo.read("main").group


def walk(group, prefix=""):
    """Every node under ``group``, as ``{path: "Group" | "Array"}``."""
    found = {}
    for name, node in group.members():
        path = f"{prefix}/{name}"
        found[path] = type(node).__name__
        if isinstance(node, zarr.Group):
            found.update(walk(node, path))
    return found


def test_each_level_is_an_array_not_a_group(written):
    """The regression guard: the level is the array's *name*, not a group above
    it. Writing to ``multiscales/{level}`` puts the array at
    ``multiscales/{level}/{level}``, one node below everything that names it."""
    hierarchy = walk(written)

    for level in range(len(SIZES)):
        path = f"/{GRANULE}/B04/{MULTISCALES_GROUP}/{level}"
        assert hierarchy.get(path) == "Array", hierarchy
    # Nothing below the arrays: no /{level}/{level}.
    assert not [p for p in hierarchy if p.count("/") > 4]


def test_layout_paths_resolve_to_the_arrays_from_where_the_attributes_live(written):
    """A layout entry names a path relative to the group carrying it. If that
    path does not reach an array, the pyramid declaration points at nothing."""
    band = written[f"{GRANULE}/B04"]
    layout = dict(band.attrs)["multiscales"]["layout"]

    assert len(layout) == len(SIZES)
    for index, entry in enumerate(layout):
        resolved = band[entry["asset"]]
        assert isinstance(resolved, zarr.Array), entry
        height, width = SIZES[index][1], SIZES[index][0]
        assert resolved.shape == (height, width)
        if "derived_from" in entry:
            assert isinstance(band[entry["derived_from"]], zarr.Array), entry


def test_every_band_gets_its_own_pyramid_and_attributes(written):
    for band_name in ("B04", "B03"):
        band = written[f"{GRANULE}/{band_name}"]
        attrs = dict(band.attrs)
        assert attrs["proj:code"] == "EPSG:32620"
        assert len(attrs["multiscales"]["layout"]) == len(SIZES)
        assert isinstance(band[f"{MULTISCALES_GROUP}/0"], zarr.Array)


def test_batch_path_writes_the_same_hierarchy(tmp_path):
    """The seams reach through ``ingest_batch``, so the batch path -- not just a
    hand-driven transaction -- is covered by the assertions above."""
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)
    rows = pa.Table.from_pylist([row(bands=("B04",))], schema=SOURCE.as_arrow())

    result = ingest_batch(
        repo,
        rows,
        registry=None,
        bands=("B04",),
        writer=functools.partial(write_granule, **seams()),
    )

    assert result.committed == [GRANULE]
    snap = repo.read("main")
    assert snap.table("granules").scan().to_arrow()["array_path"].to_pylist() == [
        f"/{GRANULE}"
    ]
    assert isinstance(
        snap.group[f"{GRANULE}/B04/{MULTISCALES_GROUP}/0"], zarr.Array
    )
