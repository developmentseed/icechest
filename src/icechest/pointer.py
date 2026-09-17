"""The Iceberg table pointer, carried in Icechunk commit metadata.

This is the heart of the design. Iceberg normally resolves "what is the current
version of this table?" through a catalog service. Here the Icechunk snapshot
resolves it: every commit carries the location of the Iceberg ``metadata.json``
that describes the arrays in that same commit.

Because the pointer travels *in the commit*, one Icechunk commit publishes the
new array chunks and the new table version together or not at all, and an
Icechunk tag pins both.

Commit metadata is the right home for it. It is per-snapshot and supplied at
commit time, so concurrent writers never contend for it and rebase only has to
reconcile array chunks. It is also known at exactly the right moment: after any
conflict-driven replay onto another writer's table version.
"""

from __future__ import annotations

from typing import Any

import icechunk

#: Key under which the table -> metadata.json mapping lives in commit metadata.
POINTER_KEY = "iceberg:table_versions"


def read_pointers(repo: icechunk.Repository, snapshot_id: str) -> dict[str, str]:
    """Return the ``{table_name: metadata_location}`` map recorded at a snapshot."""
    metadata: dict[str, Any] = repo.lookup_snapshot(snapshot_id).metadata or {}
    return dict(metadata.get(POINTER_KEY) or {})


def read_pointers_at_branch(repo: icechunk.Repository, branch: str) -> dict[str, str]:
    """Return the pointer map at the current tip of ``branch``."""
    return read_pointers(repo, repo.lookup_branch(branch))


def read_pointers_at_tag(repo: icechunk.Repository, tag: str) -> dict[str, str]:
    """Return the pointer map pinned by ``tag``.

    The tag pins the arrays and the table version describing them in lockstep,
    which is the whole point of the design.
    """
    return read_pointers(repo, repo.lookup_tag(tag))


def commit_metadata(pointers: dict[str, str]) -> dict[str, Any]:
    """Wrap a pointer map for passing to ``Session.commit(metadata=...)``."""
    return {POINTER_KEY: dict(pointers)}
