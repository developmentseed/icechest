"""Tables loaded from a snapshot are for reading."""

from __future__ import annotations

import pytest

from tests.helpers import seed


def test_snapshot_table_refuses_writes(repo):
    """Writing here would orphan a metadata.json nothing ever references."""
    seed(repo, "g1", "g2")
    table = repo.read("main").table("granules")

    with pytest.raises(NotImplementedError, match="read-only"):
        table.delete("granule_id == 'g1'")

    rows = repo.read("main").table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g1", "g2"}


def test_snapshot_table_still_reads(repo):
    seed(repo, "g1")
    assert repo.read("main").table("granules").scan().to_arrow().num_rows == 1
