"""Virtualization, driven through its injected seam so no COG is needed."""

from __future__ import annotations

import icechunk
import numpy as np
import pytest
import virtualizarr  # noqa: F401 -- registers the xr.Dataset.vz accessor
import xarray as xr
import zarr

from icechest.demo.conventions import MULTISCALES_GROUP
from icechest.demo.virtualize import (
    GranuleError,
    virtual_granule,
    write_granule_to_store,
)

HREF = (
    "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/"
    "HLS.L30.T20JKP.2026004T142004.v2.0/HLS.L30.T20JKP.2026004T142004.v2.0.B04.tif"
)


def build_tree(sizes):
    """What ``open_virtual_datatree(url, registry, VirtualTIFF(ifd=None,
    ifd_layout="nested"))`` returns: one child group per IFD, keyed by the
    IFD's index, each holding one array named after that same index with
    dims ``(y, x)`` -- a TIFF reports width before height, so ``sizes`` here
    is ``(width, height)`` pairs, matching that reporting order.
    """
    root = xr.DataTree()
    for level, (width, height) in enumerate(sizes):
        dataset = xr.Dataset({str(level): (("y", "x"), np.zeros((height, width)))})
        root[str(level)] = xr.DataTree(dataset=dataset)
    return root


def row(shape=(3660, 3660), bands=("B04",)):
    return {
        "id": "HLS.L30.T20JKP.2026004T142004.v2.0",
        "proj:epsg": 32620,
        "proj:shape": list(shape),
        "proj:transform": [30.0, 0.0, 199980.0, 0.0, -30.0, -3099960.0, 0.0, 0.0, 1.0],
        "assets": {band: {"href": HREF} for band in bands},
    }


def test_opens_one_pyramid_call_per_band():
    """``VirtualTIFF(ifd=None)`` parses every IFD in one pass, so opening a
    band's pyramid must cost one call, not one call per level."""
    calls = []

    def open_pyramid(url, registry):
        calls.append(url)
        return build_tree([(3660, 3660), (1830, 1830), (915, 915)])

    arrays = virtual_granule(
        row(),
        registry=None,
        bands=("B04",),
        open_pyramid=open_pyramid,
    )

    assert len(calls) == 1
    assert calls[0].startswith("s3://lp-prod-protected/")
    assert arrays.levels == {"B04": 3}
    assert arrays.shape == (3660, 3660)
    assert set(arrays.datasets["B04"]) == {0, 1, 2}
    for level, dataset in arrays.datasets["B04"].items():
        assert list(dataset.data_vars) == [str(level)]


def test_opens_exactly_one_pyramid_call_per_band_across_multiple_bands():
    calls = []

    def open_pyramid(url, registry):
        calls.append(url)
        return build_tree([(3660, 3660)])

    virtual_granule(
        row(bands=("B04", "B03")),
        registry=None,
        bands=("B04", "B03"),
        open_pyramid=open_pyramid,
    )

    assert len(calls) == 2


def test_shape_disagreement_fails_the_granule():
    """The record and the data contradicting each other is the one thing this
    store exists to make impossible, so it must not be published."""
    with pytest.raises(GranuleError, match="proj:shape"):
        virtual_granule(
            row(shape=(1830, 1830)),
            registry=None,
            bands=("B04",),
            open_pyramid=lambda url, registry: build_tree([(3660, 3660)]),
        )


def test_shape_is_rows_by_columns_not_width_by_height():
    """A TIFF reports width first, proj:shape reports rows first. Squares hide
    a transposition, so this case is deliberately not square."""
    arrays = virtual_granule(
        row(shape=(500, 800)),  # 500 rows, 800 columns
        registry=None,
        bands=("B04",),
        open_pyramid=lambda url, registry: build_tree([(800, 500)]),  # w=800, h=500
    )
    assert arrays.shape == (500, 800)


def test_granule_with_no_usable_assets_fails():
    with pytest.raises(GranuleError, match="no assets"):
        virtual_granule(
            {"id": "x", "proj:shape": [1, 1], "assets": {}},
            registry=None,
            bands=("B04",),
            open_pyramid=lambda url, registry: build_tree([(1, 1)]),
        )


def test_granule_with_no_readable_levels_fails():
    """An empty pyramid -- e.g. a corrupt or truncated COG -- must fail the
    granule rather than silently publishing zero levels."""
    with pytest.raises(GranuleError, match="no readable levels"):
        virtual_granule(
            row(),
            registry=None,
            bands=("B04",),
            open_pyramid=lambda url, registry: build_tree([]),
        )


def test_access_mode_chooses_the_url_the_headers_are_read_through():
    """Which endpoint a COG is parsed through is the ingesting machine's
    business; it must not change what ends up in the manifest."""
    opened = []

    arrays = virtual_granule(
        row(),
        registry=None,
        bands=("B04",),
        access="https",
        open_pyramid=lambda url, registry: opened.append(url)
        or build_tree([(3660, 3660)]),
    )

    assert arrays.levels == {"B04": 1}
    assert all(
        url.startswith("https://data.lpdaac.earthdatacloud.nasa.gov/") for url in opened
    )


def test_writes_into_a_bare_store(tmp_path):
    """A ForkSession has a ``.store`` but no transaction behind it: the whole
    point of splitting the store-level write out of ``write_granule`` is that
    it must not need one."""
    repo = icechunk.Repository.open_or_create(
        icechunk.local_filesystem_storage(str(tmp_path))
    )
    session = repo.writable_session("main")

    path = write_granule_to_store(
        session.store,
        row(),
        registry=None,
        bands=("B04",),
        open_pyramid=lambda url, registry: build_tree([(3660, 3660), (1830, 1830)]),
    )

    assert path == "/HLS.L30.T20JKP.2026004T142004.v2.0"
    group = zarr.open_group(session.store, path=f"{path}/B04", mode="r")
    names = [c["name"] for c in group.attrs["zarr_conventions"]]
    assert "multiscales" in names
    for level in (0, 1):
        array = zarr.open_array(
            session.store, path=f"{path}/B04/{MULTISCALES_GROUP}/{level}/B04"
        )
        assert array is not None
