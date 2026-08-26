"""What the conflict window does and does not consider a conflict."""

from __future__ import annotations

import pytest
from pyiceberg.exceptions import ValidationException
from pyiceberg.expressions.parser import parse
from pyiceberg.table.update.validate import _added_data_files

from icechest.validation import conflicting_adds
from tests.helpers import granules, seed


@pytest.fixture
def two_snapshots(repo):
    """A base version, then a disjoint append by another writer.

    Returns the table loaded at the tip, plus both snapshot ids.
    """
    seed(repo, "g1", "g2")
    base_id = repo.read("main").table("granules").metadata.current_snapshot_id

    with repo.transaction("main", "ben ingests g4") as tx:
        tx.append("granules", granules("g4"))

    table = repo.read("main").table("granules")
    return table, base_id, table.metadata.current_snapshot_id


def test_reports_the_winners_matching_add(two_snapshots):
    table, base_id, tip_id = two_snapshots

    ids = conflicting_adds(
        table,
        base_snapshot_id=base_id,
        tip_snapshot_id=tip_id,
        predicate="granule_id == 'g4'",
    )

    assert ids == [tip_id]


def test_disjoint_add_is_not_a_conflict(two_snapshots):
    table, base_id, tip_id = two_snapshots

    ids = conflicting_adds(
        table,
        base_snapshot_id=base_id,
        tip_snapshot_id=tip_id,
        predicate="granule_id == 'nothing-matches-this'",
    )

    assert ids == []


def test_base_snapshot_is_outside_the_window(two_snapshots):
    """The window is (base, tip], not [base, tip].

    The rows a delete targets were usually added by the base snapshot itself.
    Counting those would make every delete conflict with itself, so the base
    snapshot's own additions must not be reported.
    """
    table, base_id, tip_id = two_snapshots

    ids = conflicting_adds(
        table,
        base_snapshot_id=base_id,
        tip_snapshot_id=tip_id,
        predicate="granule_id == 'g1'",  # added by the base snapshot
    )

    assert ids == []


def test_identical_endpoints_are_never_a_conflict(two_snapshots):
    table, _, tip_id = two_snapshots

    ids = conflicting_adds(
        table,
        base_snapshot_id=tip_id,
        tip_snapshot_id=tip_id,
        predicate="granule_id == 'g4'",
    )

    assert ids == []


def test_upstream_argument_orientation_is_inverted(two_snapshots):
    """Pin the trap our wrapper exists to hide.

    PyIceberg documents `starting_snapshot` as the snapshot current when the
    operation began, but the traversal walks ancestors *of* it, so it must be
    the newest snapshot. If a future release fixes the naming, this test fails
    and `conflicting_adds` needs its arguments swapped.
    """
    table, base_id, tip_id = two_snapshots
    base = table.metadata.snapshot_by_id(base_id)
    tip = table.metadata.snapshot_by_id(tip_id)

    with pytest.raises(ValidationException, match="No matching snapshot"):
        list(
            _added_data_files(
                table=table,
                starting_snapshot=base,  # the documented reading
                data_filter=parse("granule_id == 'g4'"),
                partition_set=None,
                parent_snapshot=tip,
            )
        )
