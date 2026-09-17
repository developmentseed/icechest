"""The three URL forms an asset takes, and which is used where.

A granule's COG is one object, but it is named three ways:

* the record's own **https** href, which LP DAAC serves anywhere;
* the **s3** URL for the same object, which LP DAAC serves only inside
  us-west-2 but far faster;
* a **vcc** reference, relative to a named Icechunk virtual chunk container.

The first two are read-time choices, made by whichever machine happens to be
parsing a COG. Only the third is written into a manifest, and that is the point:
a relative reference carries no endpoint, so a reader resolves it against
whatever their own container is configured for. The same store is then readable
from a laptop over https and from us-west-2 over s3, without rewriting a thing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

LPDAAC_HTTPS_PREFIX = "https://data.lpdaac.earthdatacloud.nasa.gov/"
LPDAAC_S3_PREFIX = "s3://lp-prod-protected/"
LPDAAC_HTTPS_ASSET_PREFIX = f"{LPDAAC_HTTPS_PREFIX}lp-prod-protected/"

#: The container's name, which is what makes a ``vcc://`` reference resolvable.
#: It has to match the name the store declares; see ``demo.credentials``.
CONTAINER_NAME = "lpdaac"

#: Every asset the STAC records carry except the thumbnail, which is a browse
#: image rather than data.
DEFAULT_BANDS: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B09",
    "B10",
    "B11",
    "Fmask",
    "SZA",
    "SAA",
    "VZA",
    "VAA",
)

ACCESS_MODES = ("s3", "https")


def asset_key(href: str) -> str:
    """The object's key within the protected bucket.

    The one part that varies between the three URL forms, so every rewrite
    goes through here rather than slicing prefixes in three places.
    """
    if not href.startswith(LPDAAC_HTTPS_PREFIX):
        raise ValueError(f"not an LP DAAC distribution href: {href!r}")
    if not href.startswith(LPDAAC_HTTPS_ASSET_PREFIX):
        raise ValueError(
            f"{href!r} is not in the lp-prod-protected bucket, which is the only "
            "one this store declares a virtual chunk container for"
        )
    return href[len(LPDAAC_HTTPS_ASSET_PREFIX) :]


def to_s3_url(href: str) -> str:
    """The in-region read URL for an asset."""
    return f"{LPDAAC_S3_PREFIX}{asset_key(href)}"


def to_vcc_url(href: str) -> str:
    """The reference we record: relative to the container, endpoint-free.

    This is the only form that reaches a manifest. Writing an absolute URL
    would bake one endpoint into the data and make the store readable only
    from where it was written.
    """
    return f"vcc://{CONTAINER_NAME}/{asset_key(href)}"


def asset_urls(
    row: Mapping[str, Any],
    bands: Sequence[str] = DEFAULT_BANDS,
    access: str = "s3",
) -> dict[str, str]:
    """Map each requested band to a URL this machine can actually read.

    ``access`` is an operational choice, not a durable one: it decides how the
    COG headers are fetched while building the references, and has no bearing on
    what those references say.

    A band missing from one granule is normal and not worth failing over; a
    granule with no usable assets at all is caught downstream.
    """
    if access not in ACCESS_MODES:
        raise ValueError(
            f"unknown access mode {access!r}; expected one of {ACCESS_MODES}"
        )

    assets = row.get("assets") or {}
    urls = {}
    for band in bands:
        asset = assets.get(band)
        if asset and asset.get("href"):
            href = asset["href"]
            asset_key(href)  # refuse anything outside the container's bucket
            urls[band] = to_s3_url(href) if access == "s3" else href
    return urls
