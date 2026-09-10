"""Building a granule's virtual arrays and writing them into a transaction.

The two network-touching operations -- reading a COG header and opening a
virtual dataset over one of its IFDs -- are injected, so the orchestration around
them can be exercised without a COG.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import zarr

from icechest.demo.assets import DEFAULT_BANDS, asset_urls, to_vcc_url
from icechest.demo.conventions import MULTISCALES_GROUP, granule_attrs
from icechest.demo.tiff import HEADER_BYTES, parse_ifds


class GranuleError(Exception):
    """This granule cannot be published; the batch skips it and records why."""


@dataclass
class GranuleArrays:
    """Per-band virtual datasets, keyed by resolution level."""

    datasets: dict[str, dict[int, Any]]
    levels: dict[str, int]
    shape: tuple[int, int]


def read_header(url: str, registry: Any) -> bytes:
    """Fetch enough of a COG to cover its whole IFD chain."""
    import obstore

    store, path = registry.resolve(url)
    return bytes(obstore.get_range(store, path, start=0, end=HEADER_BYTES))


def _open_level(url: str, registry: Any, ifd: int) -> Any:
    from virtual_tiff import VirtualTIFF
    from virtualizarr import open_virtual_dataset

    return open_virtual_dataset(url=url, registry=registry, parser=VirtualTIFF(ifd=ifd))


def virtual_granule(
    row: Mapping[str, Any],
    *,
    registry: Any,
    bands: Sequence[str] = DEFAULT_BANDS,
    access: str = "s3",
    opener: Callable[[str, Any, int], Any] = _open_level,
    header_reader: Callable[[str, Any], bytes] = read_header,
) -> GranuleArrays:
    """Open every resolution level of every requested asset.

    The level count comes from each asset's own IFD chain rather than being
    assumed, because Fmask and the angle bands need not match the spectral
    bands' pyramid depth.
    """
    urls = asset_urls(row, bands, access=access)
    if not urls:
        raise GranuleError(f"{row['id']}: no assets among {tuple(bands)}")

    datasets: dict[str, dict[int, Any]] = {}
    levels: dict[str, int] = {}
    shape: tuple[int, int] | None = None

    for band, url in urls.items():
        sizes = parse_ifds(header_reader(url, registry))
        if not sizes:
            raise GranuleError(f"{row['id']}: no readable IFDs in {band}")
        datasets[band] = {ifd: opener(url, registry, ifd) for ifd in range(len(sizes))}
        levels[band] = len(sizes)
        if shape is None:
            shape = (sizes[0][1], sizes[0][0])  # (rows, cols) from (width, height)

    expected = (int(row["proj:shape"][0]), int(row["proj:shape"][1]))
    if shape != expected:
        raise GranuleError(
            f"{row['id']}: proj:shape {expected} disagrees with the COG's {shape}"
        )
    return GranuleArrays(datasets=datasets, levels=levels, shape=shape)


def write_granule(
    tx: Any,
    row: Mapping[str, Any],
    *,
    registry: Any,
    bands: Sequence[str] = DEFAULT_BANDS,
    access: str = "s3",
    opener: Callable[[str, Any, int], Any] = _open_level,
    header_reader: Callable[[str, Any], bytes] = read_header,
) -> str:
    """Stage one granule's arrays and conventions, returning its group path.

    Writes into the transaction's own session, so the references are staged
    alongside the table pointer and published by the same commit.

    ``VirtualTIFF(ifd=n)`` yields a single-variable dataset whose variable is
    named ``str(n)``, and ``to_icechunk`` writes variables *inside* the group it
    is given -- so every level goes into one shared ``multiscales`` group and
    the variable name supplies the level. Naming the level in the group path as
    well would bury each array one node deeper than the layout declares it.

    The seams are forwarded from :func:`virtual_granule` so this whole path --
    the nesting and the attribute placement included -- can be driven without a
    COG.
    """
    arrays = virtual_granule(
        row,
        registry=registry,
        bands=bands,
        access=access,
        opener=opener,
        header_reader=header_reader,
    )
    granule_id = row["id"]
    for band, per_level in arrays.datasets.items():
        # The manifest records the reference, not the URL we happened to read
        # through: relative to the container's name, so a reader resolves it
        # against their own endpoint. Every chunk of a level points at the one
        # asset, so a single replacement path is exactly right.
        reference = to_vcc_url(row["assets"][band]["href"])
        for _level, dataset in sorted(per_level.items()):
            # validate_containers is off because that check matches container
            # prefixes literally and does not understand the vcc:// scheme; it
            # would reject every relative reference. What it was guarding
            # against -- a reference no container can resolve -- is instead
            # ruled out by construction: every href goes through asset_key,
            # which refuses anything outside the container's bucket, before a
            # single byte is staged.
            dataset.vz.rename_paths(reference).vz.to_icechunk(
                store=tx.session.store,
                group=f"/{granule_id}/{band}/{MULTISCALES_GROUP}",
                mode="a",
                validate_containers=False,
            )
        # The attributes live on the band group, one above the levels, so the
        # layout's paths are prefixed with the multiscales group's name.
        group = zarr.open_group(
            tx.session.store, path=f"/{granule_id}/{band}", mode="a"
        )
        group.attrs.update(
            granule_attrs(
                epsg=row["proj:epsg"],
                shape=row["proj:shape"],
                transform=row["proj:transform"],
                levels=arrays.levels[band],
            )
        )
    return f"/{granule_id}"
