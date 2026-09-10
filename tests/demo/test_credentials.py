"""Earthdata Basic Auth: proactive, not reactive.

``lpdaac_credentials`` used to rely on ``urllib.request.HTTPBasicAuthHandler``,
which only attaches credentials after a 401 challenge. URS's OAuth
``/authorize`` page never sends one -- it returns the login form directly --
so that handler never fired and the "credentials" call quietly got HTML back
instead of JSON. None of that showed up here, because it only breaks against
the real service; these tests pin the fix down with no network at all, so a
regression shows up here instead of on someone else's infrastructure months
from now.
"""

from __future__ import annotations

import base64
import re
import urllib.request

import pytest

from icechest.demo import credentials


class _FakeNetrc:
    """A ``netrc.netrc`` stand-in with a fixed set of entries, no file I/O."""

    def __init__(self, entries):
        self._entries = entries

    def authenticators(self, host):
        return self._entries.get(host)


def _patch_netrc(monkeypatch, entries):
    monkeypatch.setattr(
        credentials.netrc, "netrc", lambda *a, **kw: _FakeNetrc(entries)
    )


def test_auth_header_is_the_correct_basic_encoding(monkeypatch):
    _patch_netrc(
        monkeypatch, {credentials.EARTHDATA_HOST: ("someuser", None, "somepass")}
    )

    header = credentials._earthdata_auth_header()

    assert header == "Basic " + base64.b64encode(b"someuser:somepass").decode()


def test_missing_netrc_entry_raises_a_clear_error(monkeypatch):
    _patch_netrc(monkeypatch, {})

    with pytest.raises(RuntimeError, match=re.escape(credentials.EARTHDATA_HOST)):
        credentials._earthdata_auth_header()


def test_auth_header_is_attached_to_the_first_request_not_after_a_401(monkeypatch):
    """The whole bug, pinned down: the header must be on the request the
    opener sends *first* -- there is no second, challenge-response request in
    this flow for a reactive handler to answer."""
    _patch_netrc(
        monkeypatch, {credentials.EARTHDATA_HOST: ("someuser", None, "somepass")}
    )
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return (
                b'{"accessKeyId": "AKIA", "secretAccessKey": "secret", '
                b'"sessionToken": "token", "expiration": "2026-01-01T00:00:00+00:00"}'
            )

    class _FakeOpener:
        def open(self, request, timeout=None):
            captured["request"] = request
            return _FakeResponse()

    monkeypatch.setattr(credentials, "_earthdata_opener", lambda: _FakeOpener())

    creds = credentials.lpdaac_credentials()

    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    assert request.get_header("Authorization") == (
        "Basic " + base64.b64encode(b"someuser:somepass").decode()
    )
    # And the round trip actually produced credentials, confirming the fake
    # opener's response was consumed the same way the real one would be.
    assert creds.access_key_id == "AKIA"


def test_a_missing_netrc_reads_like_a_missing_entry(monkeypatch):
    """The two ways of not having Earthdata credentials are the same problem to
    whoever is running this, so they raise the same kind of error -- rather than
    a bare FileNotFoundError from one branch and a clear message from the other."""

    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "~/.netrc")

    monkeypatch.setattr(credentials.netrc, "netrc", missing)
    with pytest.raises(RuntimeError, match="no ~/.netrc"):
        credentials._earthdata_auth_header()
