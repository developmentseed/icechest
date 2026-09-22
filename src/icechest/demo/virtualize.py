"""Building a granule's virtual arrays and writing them into a transaction.

The network-touching operation -- opening every resolution level of a band's
COG -- is injected, so the orchestration around it can be exercised without a
COG.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import zarr

from icechest.demo.assets import DEFAULT_BANDS, asset_urls, to_vcc_url
from icechest.demo.conventions import MULTISCALES_GROUP, granule_attrs
from icechest.demo.tiff import HEADER_BYTES


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


def _open_pyramid(url: str, registry: Any) -> Any:
    """Open every resolution level of one band's COG in a single read.

    ``VirtualTIFF(ifd=None)`` parses the whole IFD chain in one pass over the
    file instead of the parser reopening it once per level. ``ifd_layout``
    must be ``"nested"``: ``open_virtual_dataset`` with the same parser
    either returns an empty top-level dataset (``"nested"``) or raises on
    colliding y/x dimensions across sibling levels (``"flat"``), so only
    ``open_virtual_datatree`` can represent a pyramid this way -- one child
    group per IFD, holding one array named after that IFD's index.
    """
    from virtual_tiff import VirtualTIFF
    from virtualizarr import open_virtual_datatree

    return open_virtual_datatree(
        url=url,
        registry=registry,
        parser=VirtualTIFF(ifd=None, ifd_layout="nested"),
    )


def virtual_granule(
    row: Mapping[str, Any],
    *,
    registry: Any,
    bands: Sequence[str] = DEFAULT_BANDS,
    access: str = "s3",
    open_pyramid: Callable[[str, Any], Any] = _open_pyramid,
) -> GranuleArrays:
    """Open every resolution level of every requested asset.

    The level count comes from each asset's own pyramid rather than being
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
        tree = open_pyramid(url, registry)
        per_level = {int(key): node.to_dataset() for key, node in tree.children.items()}
        if not per_level:
            raise GranuleError(f"{row['id']}: no readable levels in {band}")
        datasets[band] = per_level
        levels[band] = len(per_level)
        if shape is None:
            level0 = per_level[0]
            shape = (int(level0.sizes["y"]), int(level0.sizes["x"]))

    expected = (int(row["proj:shape"][0]), int(row["proj:shape"][1]))
    if shape != expected:
        raise GranuleError(
            f"{row['id']}: proj:shape {expected} disagrees with the COG's {shape}"
        )
    return GranuleArrays(datasets=datasets, levels=levels, shape=shape)


def write_granule_to_store(
    store: Any,
    row: Mapping[str, Any],
    *,
    registry: Any,
    bands: Sequence[str] = DEFAULT_BANDS,
    access: str = "s3",
    open_pyramid: Callable[[str, Any], Any] = _open_pyramid,
) -> str:
    """Stage one granule's arrays and conventions into ``store``.

    Split out of :func:`write_granule` so a backfill worker can write into a
    ``ForkSession.store``, which has no transaction behind it -- only
    :func:`write_granule` needs one, to stage the references alongside the
    table pointer and publish both by the same commit.

    Each level is a group of its own holding one array, because levels differ
    in y and x and dimensions of one name must agree within a node: as sibling
    arrays in a shared group they collide, and neither ``xr.open_dataset`` nor
    ``xr.open_datatree`` can open the pyramid.

    Each level's dataset carries one variable named after the IFD it came
    from (``str(n)``), and ``to_icechunk`` writes variables *inside* the group
    it is given. Left alone that puts an array called ``"1"`` inside the group
    already called ``1``, and names the data after the IFD it was read from,
    which says nothing about what it is -- so the variable is renamed to the
    band first.

    The seam is forwarded from :func:`virtual_granule` so this whole path --
    the nesting and the attribute placement included -- can be driven without a
    COG.
    """
    arrays = virtual_granule(
        row,
        registry=registry,
        bands=bands,
        access=access,
        open_pyramid=open_pyramid,
    )
    granule_id = row["id"]
    for band, per_level in arrays.datasets.items():
        # The manifest records the reference, not the URL we happened to read
        # through: relative to the container's name, so a reader resolves it
        # against their own endpoint. Every chunk of a level points at the one
        # asset, so a single replacement path is exactly right.
        reference = to_vcc_url(row["assets"][band]["href"])
        for level, dataset in sorted(per_level.items()):
            variables = list(dataset.data_vars)
            if len(variables) != 1:
                raise GranuleError(
                    f"{row['id']}: expected one variable per level in {band}, "
                    f"got {variables}"
                )
            leveled = dataset.rename({variables[0]: band})
            # validate_containers is off because that check matches container
            # prefixes literally and does not understand the vcc:// scheme; it
            # would reject every relative reference. What it was guarding
            # against -- a reference no container can resolve -- is instead
            # ruled out by construction: every href goes through asset_key,
            # which refuses anything outside the container's bucket, before a
            # single byte is staged.
            leveled.vz.rename_paths(reference).vz.to_icechunk(
                store=store,
                group=f"/{granule_id}/{band}/{MULTISCALES_GROUP}/{level}",
                mode="a",
                validate_containers=False,
            )
        # The attributes live on the band group, one above the levels, so the
        # layout's paths are prefixed with the multiscales group's name.
        group = zarr.open_group(store, path=f"/{granule_id}/{band}", mode="a")
        group.attrs.update(
            granule_attrs(
                epsg=row["proj:epsg"],
                shape=row["proj:shape"],
                transform=row["proj:transform"],
                levels=arrays.levels[band],
                array_name=band,
            )
        )
    return f"/{granule_id}"


def write_granule(
    tx: Any,
    row: Mapping[str, Any],
    *,
    registry: Any,
    bands: Sequence[str] = DEFAULT_BANDS,
    access: str = "s3",
    open_pyramid: Callable[[str, Any], Any] = _open_pyramid,
) -> str:
    """Stage one granule's arrays and conventions, returning its group path.

    Writes into the transaction's own session, so the references are staged
    alongside the table pointer and published by the same commit. The actual
    work is :func:`write_granule_to_store`; this just supplies the store a
    transaction carries.
    """
    return write_granule_to_store(
        tx.session.store,
        row,
        registry=registry,
        bands=bands,
        access=access,
        open_pyramid=open_pyramid,
    )
