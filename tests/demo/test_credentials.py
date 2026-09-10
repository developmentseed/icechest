"""Credentials and the container a reference is resolved against.

Token lifecycle and the per-DAAC credential exchange belong to
``earthaccess-auth``; what is tested here is our side of the seam -- that the
container is named and pointed correctly for each access mode, and that what
the library hands back is mapped onto what icechunk expects.
"""

from __future__ import annotations

import pickle
from datetime import UTC, datetime

import pytest

from icechest.demo import credentials
from icechest.demo.assets import (
    CONTAINER_NAME,
    LPDAAC_HTTPS_ASSET_PREFIX,
    LPDAAC_S3_PREFIX,
)


class FakeS3Credentials:
    access_key_id = "AKIA-fake"
    secret_access_key = "secret-fake"
    session_token = "token-fake"
    expires_at = datetime(2026, 9, 10, 18, 0, tzinfo=UTC)


class FakeManager:
    def __init__(self):
        self.asked_for = None

    def get_bucket_credentials(self, bucket_or_url):
        self.asked_for = bucket_or_url
        return FakeS3Credentials()


def test_container_is_named_so_references_can_be_relative():
    """A ``vcc://`` reference names the container; without a name there is
    nothing for it to resolve against."""
    for access in ("s3", "https"):
        container = credentials.virtual_chunk_container(access, token="t")
        assert container.name == CONTAINER_NAME


def test_s3_container_points_at_the_bucket():
    container = credentials.virtual_chunk_container("s3")
    assert container.url_prefix == LPDAAC_S3_PREFIX


def test_https_container_points_at_the_distribution_endpoint():
    """The same objects, reachable from outside us-west-2."""
    container = credentials.virtual_chunk_container("https", token="t")
    assert container.url_prefix == LPDAAC_HTTPS_ASSET_PREFIX


def test_https_container_needs_a_token():
    """icechunk's HTTP store takes static headers only, so the bearer has to be
    in hand when the container is built -- there is no callback to fetch it
    later, and a container without one would 401 on every chunk."""
    with pytest.raises(ValueError, match="token"):
        credentials.virtual_chunk_container("https")


def test_unknown_access_mode_is_refused():
    with pytest.raises(ValueError, match="access"):
        credentials.virtual_chunk_container("ftp")


def test_credential_callable_is_picklable():
    """icechunk requires it: refreshable credentials may cross a process."""
    assert (
        pickle.loads(pickle.dumps(credentials.lpdaac_credentials))
        is credentials.lpdaac_credentials
    )


def test_credentials_are_mapped_onto_icechunks_fields(monkeypatch):
    """The library's expiry drives icechunk's refresh, so a dropped
    ``expires_at`` would leave icechunk holding a credential past its life."""
    manager = FakeManager()
    monkeypatch.setattr(credentials, "default_manager", lambda: manager)

    result = credentials.lpdaac_credentials()

    assert manager.asked_for == LPDAAC_S3_PREFIX
    assert result.access_key_id == FakeS3Credentials.access_key_id
    assert result.secret_access_key == FakeS3Credentials.secret_access_key
    assert result.session_token == FakeS3Credentials.session_token
    assert result.expires_after == FakeS3Credentials.expires_at
