"""Asset href rewriting: the URL we record is the URL every reader resolves."""

from __future__ import annotations

import pytest

from icechest.demo.assets import DEFAULT_BANDS, asset_urls, to_s3_url

HREF = (
    "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/"
    "HLS.L30.T20JKP.2026004T142004.v2.0/HLS.L30.T20JKP.2026004T142004.v2.0.B04.tif"
)
S3 = (
    "s3://lp-prod-protected/HLSL30.020/"
    "HLS.L30.T20JKP.2026004T142004.v2.0/HLS.L30.T20JKP.2026004T142004.v2.0.B04.tif"
)


def test_rewrites_lpdaac_href_to_s3():
    assert to_s3_url(HREF) == S3


def test_unexpected_host_is_refused():
    with pytest.raises(ValueError, match="not an LP DAAC"):
        to_s3_url("https://example.com/some/other.tif")


def test_asset_urls_maps_requested_bands():
    row = {"assets": {"B04": {"href": HREF}, "B03": {"href": HREF}}}
    assert asset_urls(row, bands=("B04",)) == {"B04": S3}


def test_missing_and_null_bands_are_skipped():
    row = {"assets": {"B04": {"href": HREF}, "B03": None, "Fmask": {"href": None}}}
    assert set(asset_urls(row, bands=("B04", "B03", "Fmask"))) == {"B04"}


def test_default_bands_are_the_fifteen_non_thumbnail_assets():
    assert "thumbnail" not in DEFAULT_BANDS
    assert len(DEFAULT_BANDS) == 15
    assert DEFAULT_BANDS[0] == "B01" and "VAA" in DEFAULT_BANDS
