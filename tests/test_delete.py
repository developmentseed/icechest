"""Deletes and overwrites: replay when provably safe, fail when not."""

from __future__ import annotations

import numpy as np
import pytest

from icechest import TableConflictError
from tests.helpers import granules, seed


def test_delete_replays_over_disjoint_append(repo):
    """Anna retracts a granule while Ben ingests an unrelated one."""
    seed(repo, "g1", "g2", "g3")

    anna = repo.transaction("main", "anna retracts g1")
    ben = repo.transaction("main", "ben ingests g4")

    anna.group["data"][50:60] = np.full(10, 2.0, "f4")
    anna.delete("granules", "granule_id == 'g1'")

    ben.group["data"][0:10] = np.ones(10, "f4")
    ben.append("granules", granules("g4"))

    ben.commit()
    anna.commit()  # conflicts, validates, replays onto Ben's version

    snap = repo.read("main")
    rows = snap.table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g2", "g3", "g4"}
    assert snap.group["data"][0] == 1.0
    assert snap.group["data"][50] == 2.0


def test_delete_fails_when_winner_adds_matching_row(repo):
    """Ben re-ingests the granule Anna is retracting: she must re-plan."""
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna retracts g1")
    ben = repo.transaction("main", "ben re-ingests g1")

    anna.group["data"][50:60] = np.full(10, 2.0, "f4")
    anna.delete("granules", "granule_id == 'g1'")

    ben.append("granules", granules("g1"))
    ben_snapshot = ben.commit()

    with pytest.raises(TableConflictError) as excinfo:
        anna.commit()

    assert excinfo.value.table == "granules"
    assert excinfo.value.snapshot_ids

    tip = repo.read("main")
    assert tip.snapshot_id == ben_snapshot  # Anna published nothing
    assert tip.group["data"][50] == 0.0  # her array write died with the delete
    assert tip.table("granules").scan().to_arrow().num_rows == 3


def test_overwrite_reingest_replays_over_disjoint_append(repo):
    """Anna corrects g1's row while Ben ingests g9."""
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna re-ingests g1")
    ben = repo.transaction("main", "ben ingests g9")

    anna.overwrite(
        "granules", granules("g1", day="2026-02-02"), "granule_id == 'g1'"
    )
    anna.group["data"][50:60] = np.full(10, 2.0, "f4")

    ben.append("granules", granules("g9"))
    ben.commit()
    anna.commit()

    rows = repo.read("main").table("granules").scan().to_arrow()
    assert sorted(rows["granule_id"].to_pylist()) == ["g1", "g2", "g9"]
    corrected = dict(
        zip(
            rows["granule_id"].to_pylist(),
            rows["datetime"].to_pylist(),
            strict=True,
        )
    )
    assert corrected["g1"].date().isoformat() == "2026-02-02"
