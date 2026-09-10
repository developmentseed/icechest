"""Earthdata credentials for the LP DAAC bucket holding the COGs.

The metadata and the assets live behind different credentials: the archive is a
MAAP bucket read with the ambient AWS profile, while the COGs need temporary
keys minted from an Earthdata login. Those keys last about an hour, so icechunk
refreshes them through a callable rather than holding a snapshot of them -- and
that callable must be picklable, which is why it is a module-level function.
"""

from __future__ import annotations

import json
import netrc
import urllib.request
from datetime import datetime
from http.cookiejar import CookieJar

import icechunk

from icechest.demo.assets import LPDAAC_S3_PREFIX

EARTHDATA_HOST = "urs.earthdata.nasa.gov"
LPDAAC_CREDENTIALS_URL = "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
LPDAAC_REGION = "us-west-2"


def _earthdata_opener() -> urllib.request.OpenerDirector:
    """An opener that can follow Earthdata's login redirect.

    The credentials endpoint bounces through URS, so both basic auth and a
    cookie jar are needed for the round trip to complete.
    """
    auth = netrc.netrc().authenticators(EARTHDATA_HOST)
    if auth is None:
        raise RuntimeError(f"no ~/.netrc entry for {EARTHDATA_HOST}")
    username, _, password = auth
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, f"https://{EARTHDATA_HOST}", username, password)
    return urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(manager),
        urllib.request.HTTPCookieProcessor(CookieJar()),
    )


def lpdaac_credentials() -> icechunk.S3StaticCredentials:
    """Mint temporary S3 credentials for the LP DAAC protected bucket."""
    with _earthdata_opener().open(LPDAAC_CREDENTIALS_URL, timeout=60) as response:
        payload = json.load(response)
    return icechunk.S3StaticCredentials(
        access_key_id=payload["accessKeyId"],
        secret_access_key=payload["secretAccessKey"],
        session_token=payload["sessionToken"],
        expires_after=datetime.fromisoformat(payload["expiration"]),
    )


def virtual_chunk_container() -> icechunk.VirtualChunkContainer:
    """Declare where this store's virtual references point."""
    return icechunk.VirtualChunkContainer(
        url_prefix=LPDAAC_S3_PREFIX,
        store=icechunk.s3_store(region=LPDAAC_REGION),
    )


def object_store_registry():
    """An obstore registry VirtualiZarr can read COG headers through.

    Separate from the icechunk container: that one is how a *reader* resolves a
    stored reference, this one is how we read the headers to create it.
    """
    import obstore
    from obspec_utils.registry import ObjectStoreRegistry

    credentials = lpdaac_credentials()
    store = obstore.store.from_url(
        LPDAAC_S3_PREFIX,
        region=LPDAAC_REGION,
        access_key_id=credentials.access_key_id,
        secret_access_key=credentials.secret_access_key,
        token=credentials.session_token,
    )
    return ObjectStoreRegistry({LPDAAC_S3_PREFIX: store})
