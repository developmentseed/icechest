"""Commit-time bookkeeping for transactions whose table work is not an append."""

from __future__ import annotations

import numpy as np
import pytest

from icechest import TableConflictError
from tests.helpers import granules, seed


def test_delete_only_transaction_commits(repo):
    """No Zarr node changes, but the pointer moved: that is a real commit."""
    seed(repo, "g1", "g2")
    before = repo.read("main").snapshot_id

    with repo.transaction("main", "retract g1") as tx:
        tx.delete("granules", "granule_id == 'g1'")

    snap = repo.read("main")
    assert snap.snapshot_id != before
    rows = snap.table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g2"}


def test_transaction_with_no_changes_raises(repo):
    """A delete matching nothing stages nothing, so nothing was written."""
    seed(repo, "g1")

    with pytest.raises(ValueError, match="made no changes"):
        with repo.transaction("main", "zero-match retract") as tx:
            tx.delete("granules", "granule_id == 'no-such-granule'")


def test_zero_match_delete_commits_alongside_other_work(repo):
    seed(repo, "g1")

    with repo.transaction("main", "ingest and prune") as tx:
        tx.group["data"][0:10] = np.ones(10, "f4")
        tx.delete("granules", "granule_id == 'no-such-granule'")

    snap = repo.read("main")
    assert snap.group["data"][0] == 1.0
    assert snap.table("granules").scan().to_arrow().num_rows == 1


def test_concurrent_identical_delete_replays_as_empty_commit(repo):
    """Two writers delete the same rows; the loser's replay stages nothing.

    Anna's intent is already satisfied on the branch once Ben's delete wins
    the race, so replaying it legitimately stages nothing. That must not be
    confused with "no changes were ever requested" -- the loser's commit()
    should still succeed and return a snapshot id, with the rows gone
    exactly once.
    """
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna retracts g1")
    ben = repo.transaction("main", "ben also retracts g1")

    anna.delete("granules", "granule_id == 'g1'")
    ben.delete("granules", "granule_id == 'g1'")

    ben.commit()
    snapshot_id = anna.commit()

    assert snapshot_id is not None
    snap = repo.read("main")
    assert snap.snapshot_id == snapshot_id
    rows = snap.table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g2"}


def test_failed_commit_discards_the_session(repo):
    """A refused replay must not leave array writes live on the session."""
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna retracts g1")
    ben = repo.transaction("main", "ben re-ingests g1")

    anna.group["data"][50:60] = np.full(10, 2.0, "f4")
    anna.delete("granules", "granule_id == 'g1'")

    ben.append("granules", granules("g1"))
    ben.commit()

    with pytest.raises(TableConflictError):
        with anna:
            pass

    assert anna.session.has_uncommitted_changes is False
