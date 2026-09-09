"""Turning a STAC record's asset hrefs into the URLs we will record.

The archive publishes LP DAAC HTTPS hrefs, but a virtual reference stores the
URL every future reader must resolve. ``s3://`` is the form that works in-region
at scale, and it is what the Icechunk virtual chunk container is declared
against, so the rewrite happens once here rather than at read time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

LPDAAC_HTTPS_PREFIX = "https://data.lpdaac.earthdatacloud.nasa.gov/"
LPDAAC_S3_PREFIX = "s3://lp-prod-protected/"

#: Every asset the STAC records carry except the thumbnail, which is a browse
#: image rather than data.
DEFAULT_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B09", "B10", "B11",
    "Fmask", "SZA", "SAA", "VZA", "VAA",
)


def to_s3_url(href: str) -> str:
    """Rewrite an LP DAAC distribution href to the S3 URL we will record.

    The result must fall under ``LPDAAC_S3_PREFIX``: that is the prefix the
    Icechunk virtual chunk container is declared against, so a URL outside it
    would be written as a reference nothing can resolve.
    """
    if not href.startswith(LPDAAC_HTTPS_PREFIX):
        raise ValueError(f"not an LP DAAC distribution href: {href!r}")
    url = "s3://" + href[len(LPDAAC_HTTPS_PREFIX) :]
    if not url.startswith(LPDAAC_S3_PREFIX):
        raise ValueError(
            f"{href!r} is not in the {LPDAAC_S3_PREFIX} bucket, which is the only "
            "prefix this store declares a virtual chunk container for"
        )
    return url


def asset_urls(
    row: Mapping[str, Any], bands: Sequence[str] = DEFAULT_BANDS
) -> dict[str, str]:
    """Map each requested band to its S3 URL, skipping assets the record lacks.

    A band missing from one granule is normal and not worth failing over; a
    granule with no usable assets at all is caught downstream.
    """
    assets = row.get("assets") or {}
    urls = {}
    for band in bands:
        asset = assets.get(band)
        if asset and asset.get("href"):
            urls[band] = to_s3_url(asset["href"])
    return urls
