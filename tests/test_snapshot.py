"""Reading a snapshot: the store handle, and how a table is resolved.

A snapshot is immutable, so everything read through it can share one
read-only session. What a reader must not share is the two halves of the
convention: the pointer in commit metadata is what resolves a table, and the
declaration on the group is consulted only to explain a pointer that is
missing.
"""

from __future__ import annotations

import pytest
import zarr

from icechest import transaction as transaction_module
from icechest.convention import TableBinding, declare
from tests.helpers import seed


def test_group_reads_through_the_snapshots_store(repo):
    seed(repo)
    snap = repo.read("main")

    assert snap.group.store is snap.store


def test_store_is_one_handle_reused_across_accesses(repo):
    """Each access used to open its own read-only session, so code that
    touched the snapshot twice quietly held two."""
    seed(repo)
    snap = repo.read("main")

    assert snap.store is snap.store


def test_loading_a_table_does_not_read_the_declaration(repo, monkeypatch):
    """The pointer is authoritative and per-snapshot. Consulting the group as
    well cost a session and an attribute parse to answer a set-membership
    question the pointer had already answered."""
    seed(repo, "g1")
    snap = repo.read("main")

    def refuse(group):
        raise AssertionError("table() read the group on the happy path")

    monkeypatch.setattr(transaction_module, "read_bindings", refuse)

    assert snap.table("granules").scan().to_arrow().num_rows == 1


def test_a_declared_table_with_no_pointer_is_reported_as_malformed(repo):
    """The convention requires both halves to land in the same commit, so a
    declaration without a pointer is a broken store -- and the reader must
    say so rather than fall back to another snapshot's version."""
    seed(repo)
    session = repo.repo.writable_session("main")
    declare(
        zarr.open_group(session.store, mode="a"),
        {"ghost": TableBinding(location="/w/ghost")},
    )
    session.commit("declare a table without pointing at a version")

    with pytest.raises(ValueError, match="no pointer"):
        repo.read("main").table("ghost")
