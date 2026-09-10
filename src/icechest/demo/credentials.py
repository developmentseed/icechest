"""Earthdata credentials, and the container a reference resolves against.

Two different credentials are in play. The metadata archive is a MAAP bucket
read with the ambient AWS profile; the COGs live in NASA's LP DAAC protected
bucket, which wants either temporary S3 keys minted from an Earthdata login or a
bearer token over https.

Token lifecycle, the login strategies and the per-DAAC credential exchange are
``earthaccess-auth``'s job rather than ours -- it is the library that knows
which endpoint mints credentials for which bucket, and how long they last.
"""

from __future__ import annotations

import icechunk
from earthaccess_auth import default_manager, login

from icechest.demo.assets import (
    ACCESS_MODES,
    CONTAINER_NAME,
    LPDAAC_HTTPS_ASSET_PREFIX,
    LPDAAC_S3_PREFIX,
)

LPDAAC_REGION = "us-west-2"


def lpdaac_credentials() -> icechunk.S3StaticCredentials:
    """Temporary S3 credentials for the LP DAAC protected bucket.

    Module-level because icechunk refreshes credentials through a callable it
    may pickle, and a closure over configuration would not survive that.
    """
    issued = default_manager().get_bucket_credentials(LPDAAC_S3_PREFIX)
    return icechunk.S3StaticCredentials(
        access_key_id=issued.access_key_id,
        secret_access_key=issued.secret_access_key,
        session_token=issued.session_token,
        expires_after=issued.expires_at,
    )


def earthdata_token() -> str:
    """A bearer token for reading assets over https."""
    return login(strategy="all").token["access_token"]


def virtual_chunk_container(
    access: str = "s3", *, token: str | None = None
) -> icechunk.VirtualChunkContainer:
    """Where this reader resolves ``vcc://lpdaac/...`` references.

    Named, which is what makes a relative reference resolvable at all: the
    manifest records only the key, and this supplies the endpoint. Two readers
    of the same store can therefore disagree about how to reach an object
    without either of them being wrong -- one in us-west-2 over s3, one
    anywhere over https.
    """
    if access not in ACCESS_MODES:
        raise ValueError(
            f"unknown access mode {access!r}; expected one of {ACCESS_MODES}"
        )

    if access == "s3":
        return icechunk.VirtualChunkContainer(
            url_prefix=LPDAAC_S3_PREFIX,
            store=icechunk.s3_store(region=LPDAAC_REGION),
            name=CONTAINER_NAME,
        )

    if token is None:
        raise ValueError(
            "https access needs a bearer token: icechunk's HTTP store takes "
            "static headers only, so there is no callback to fetch one later"
        )
    return icechunk.VirtualChunkContainer(
        url_prefix=LPDAAC_HTTPS_ASSET_PREFIX,
        store=icechunk.http_store(headers={"Authorization": f"Bearer {token}"}),
        name=CONTAINER_NAME,
    )


def container_credentials(access: str = "s3"):
    """Credentials for the container, in the form ``Repository`` wants.

    The https container carries its bearer in a static header, so it needs no
    separate credential; the s3 one refreshes hourly through a callable.
    """
    if access == "https":
        return icechunk.containers_credentials({LPDAAC_HTTPS_ASSET_PREFIX: None})
    return icechunk.containers_credentials(
        {
            LPDAAC_S3_PREFIX: icechunk.s3_refreshable_credentials(
                get_credentials=lpdaac_credentials
            )
        }
    )


def object_store_registry(access: str = "s3"):
    """An obstore registry VirtualiZarr reads COG headers through.

    Separate from the container: that one is how a reader resolves a stored
    reference, this one is how we read the headers to build it.
    """
    import obstore
    from obspec_utils.registry import ObjectStoreRegistry

    if access == "https":
        prefix = LPDAAC_HTTPS_ASSET_PREFIX
        store = obstore.store.from_url(
            prefix,
            client_options={
                "default_headers": {"Authorization": f"Bearer {earthdata_token()}"}
            },
        )
    else:
        issued = lpdaac_credentials()
        prefix = LPDAAC_S3_PREFIX
        store = obstore.store.from_url(
            prefix,
            region=LPDAAC_REGION,
            access_key_id=issued.access_key_id,
            secret_access_key=issued.secret_access_key,
            token=issued.session_token,
        )
    return ObjectStoreRegistry({prefix: store})
