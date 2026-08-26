"""Deletes and overwrites: replay when provably safe, fail when not."""

from __future__ import annotations

import numpy as np

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
