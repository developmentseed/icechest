"""Virtualization, driven through its injected seams so no COG is needed."""

from __future__ import annotations

import struct

import pytest

from icechest.demo.virtualize import GranuleError, virtual_granule

HREF = (
    "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/"
    "HLS.L30.T20JKP.2026004T142004.v2.0/HLS.L30.T20JKP.2026004T142004.v2.0.B04.tif"
)


def build_tiff(sizes):
    header = struct.pack("<2sHI", b"II", 42, 8)
    ifd_size = 2 + 2 * 12 + 4
    body = b""
    for index, (width, height) in enumerate(sizes):
        offset = 8 + index * ifd_size
        nxt = 0 if index == len(sizes) - 1 else offset + ifd_size
        body += struct.pack("<H", 2)
        body += struct.pack("<HHII", 256, 4, 1, width)
        body += struct.pack("<HHII", 257, 4, 1, height)
        body += struct.pack("<I", nxt)
    return header + body


def row(shape=(3660, 3660), bands=("B04",)):
    return {
        "id": "HLS.L30.T20JKP.2026004T142004.v2.0",
        "proj:epsg": 32620,
        "proj:shape": list(shape),
        "proj:transform": [30.0, 0.0, 199980.0, 0.0, -30.0, -3099960.0, 0.0, 0.0, 1.0],
        "assets": {band: {"href": HREF} for band in bands},
    }


def test_opens_one_dataset_per_level():
    opened = []

    def opener(url, registry, ifd):
        opened.append((url, ifd))
        return f"dataset-{ifd}"

    arrays = virtual_granule(
        row(),
        registry=None,
        bands=("B04",),
        opener=opener,
        header_reader=lambda url, registry: build_tiff(
            [(3660, 3660), (1830, 1830), (915, 915)]
        ),
    )

    assert arrays.levels == {"B04": 3}
    assert arrays.shape == (3660, 3660)
    assert arrays.datasets["B04"] == {0: "dataset-0", 1: "dataset-1", 2: "dataset-2"}
    assert [ifd for _, ifd in opened] == [0, 1, 2]
    assert opened[0][0].startswith("s3://lp-prod-protected/")


def test_shape_disagreement_fails_the_granule():
    """The record and the data contradicting each other is the one thing this
    store exists to make impossible, so it must not be published."""
    with pytest.raises(GranuleError, match="proj:shape"):
        virtual_granule(
            row(shape=(1830, 1830)),
            registry=None,
            bands=("B04",),
            opener=lambda url, registry, ifd: "dataset",
            header_reader=lambda url, registry: build_tiff([(3660, 3660)]),
        )


def test_shape_is_rows_by_columns_not_width_by_height():
    """A TIFF reports width first, proj:shape reports rows first. Squares hide
    a transposition, so this case is deliberately not square."""
    arrays = virtual_granule(
        row(shape=(500, 800)),  # 500 rows, 800 columns
        registry=None,
        bands=("B04",),
        opener=lambda url, registry, ifd: "dataset",
        header_reader=lambda url, registry: build_tiff([(800, 500)]),  # w=800, h=500
    )
    assert arrays.shape == (500, 800)


def test_granule_with_no_usable_assets_fails():
    with pytest.raises(GranuleError, match="no assets"):
        virtual_granule(
            {"id": "x", "proj:shape": [1, 1], "assets": {}},
            registry=None,
            bands=("B04",),
            opener=lambda url, registry, ifd: "dataset",
            header_reader=lambda url, registry: build_tiff([(1, 1)]),
        )
