"""Asset href rewriting: the URL we record is the URL every reader resolves."""

from __future__ import annotations

import pytest

from icechest.demo.assets import (
    CONTAINER_NAME,
    DEFAULT_BANDS,
    LPDAAC_HTTPS_ASSET_PREFIX,
    asset_urls,
    to_s3_url,
    to_vcc_url,
)

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


def test_href_outside_the_protected_bucket_is_refused():
    """Only lp-prod-protected has a virtual chunk container declared, so a URL
    anywhere else would be recorded as an unresolvable reference."""
    other = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-public/thing.tif"
    with pytest.raises(ValueError, match="lp-prod-protected"):
        to_s3_url(other)


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


VCC = (
    "vcc://lpdaac/HLSL30.020/"
    "HLS.L30.T20JKP.2026004T142004.v2.0/HLS.L30.T20JKP.2026004T142004.v2.0.B04.tif"
)
HTTPS_ASSET = HREF  # the record's own href is already the https form


def test_reference_url_is_relative_to_the_named_container():
    """What gets written into the manifest. Relative, so a reader resolves it
    against whichever endpoint their container is configured for."""
    assert to_vcc_url(HREF) == VCC


def test_reference_url_keeps_the_key_verbatim():
    """The key after the bucket is the only part that varies, and it has to
    survive untouched or the reference points somewhere else."""
    assert to_vcc_url(HREF).removeprefix(
        f"vcc://{CONTAINER_NAME}/"
    ) == HREF.removeprefix(LPDAAC_HTTPS_ASSET_PREFIX)


def test_reference_url_refuses_a_href_outside_the_container():
    other = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-public/thing.tif"
    with pytest.raises(ValueError, match="lp-prod-protected"):
        to_vcc_url(other)


def test_read_urls_default_to_s3():
    row = {"assets": {"B04": {"href": HREF}}}
    assert asset_urls(row, bands=("B04",)) == {"B04": S3}


def test_read_urls_can_be_https_for_a_reader_outside_the_region():
    """LP DAAC's S3 endpoint is us-west-2 only, so a laptop reads over HTTPS.
    Which one is used to parse a COG says nothing about what gets recorded."""
    row = {"assets": {"B04": {"href": HREF}}}
    assert asset_urls(row, bands=("B04",), access="https") == {"B04": HTTPS_ASSET}


def test_unknown_access_mode_is_refused():
    row = {"assets": {"B04": {"href": HREF}}}
    with pytest.raises(ValueError, match="access"):
        asset_urls(row, bands=("B04",), access="ftp")
