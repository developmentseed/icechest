"""Deletes and overwrites: replay when provably safe, fail when not."""

from __future__ import annotations

import numpy as np
import pytest

from icechest import TableConflictError, UnreplayableChangeError
from tests.helpers import GRANULE_SCHEMA, granules, seed


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


def test_direct_catalog_mutation_is_refused(repo):
    """The escape hatch that used to lose deletes silently now raises."""
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna deletes behind the intent API")
    ben = repo.transaction("main", "ben ingests g9")

    anna.group["data"][50:60] = np.full(10, 2.0, "f4")
    anna.catalog.load_table("granules").delete("granule_id == 'g1'")

    ben.append("granules", granules("g9"))
    ben.commit()

    with pytest.raises(UnreplayableChangeError, match="granules"):
        anna.commit()

    rows = repo.read("main").table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g1", "g2", "g9"}


def test_window_spans_every_intervening_commit(repo):
    """Two writers land before Anna retries; both are inside her window."""
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna retracts g1")
    anna.delete("granules", "granule_id == 'g1'")

    with repo.transaction("main", "carol re-ingests g1") as carol:
        carol.append("granules", granules("g1"))
    with repo.transaction("main", "ben ingests g8") as ben:
        ben.append("granules", granules("g8"))

    # Carol's matching add is not the tip -- Ben's disjoint commit is. A window
    # covering only the tip's own commit would see nothing to conflict with.
    with pytest.raises(TableConflictError):
        anna.commit()


def test_disjoint_pile_up_still_replays(repo):
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna retracts g1")
    anna.delete("granules", "granule_id == 'g1'")

    with repo.transaction("main", "ben ingests g8") as ben:
        ben.append("granules", granules("g8"))
    with repo.transaction("main", "carol ingests g9") as carol:
        carol.append("granules", granules("g9"))

    anna.commit()

    rows = repo.read("main").table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g2", "g8", "g9"}


def test_row_and_array_region_are_atomic(repo):
    """The causally-linked case: the row and the region it describes."""
    seed(repo, "g1", "g2")
    with repo.transaction("main", "fill g1's region") as tx:
        tx.group["data"][0:10] = np.ones(10, "f4")

    anna = repo.transaction("main", "anna retracts g1 and zeroes its region")
    anna.delete("granules", "granule_id == 'g1'")
    anna.group["data"][0:10] = np.zeros(10, "f4")

    with repo.transaction("main", "ben re-ingests g1") as ben:
        ben.append("granules", granules("g1"))

    with pytest.raises(TableConflictError):
        anna.commit()

    snap = repo.read("main")
    assert snap.group["data"][0] == 1.0  # the region was not zeroed
    assert snap.table("granules").scan().to_arrow().num_rows == 3


def test_delete_against_concurrently_created_table_is_refused(repo):
    """Two writers create the same table; ours must not delete from theirs."""
    with repo.transaction("main", "seed arrays") as tx:
        tx.group.create_array("data", shape=(100,), dtype="f4", chunks=(10,))

    anna = repo.transaction("main", "anna creates and prunes")
    anna.create_table("granules", GRANULE_SCHEMA)
    anna.delete("granules", "granule_id == 'g1'")

    with repo.transaction("main", "ben creates and fills") as ben:
        ben.create_table("granules", GRANULE_SCHEMA)
        ben.append("granules", granules("g1"))

    with pytest.raises(TableConflictError):
        anna.commit()

    rows = repo.read("main").table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g1"}
