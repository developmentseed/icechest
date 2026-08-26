"""The errors are part of the API: callers read them to decide how to re-plan."""

from __future__ import annotations

from icechest import IcechestError, TableConflictError, UnreplayableChangeError


def test_table_conflict_error_carries_replanning_details():
    err = TableConflictError("granules", "granule_id == 'g1'", [11, 22])

    assert err.table == "granules"
    assert err.predicate == "granule_id == 'g1'"
    assert err.snapshot_ids == [11, 22]
    assert isinstance(err, IcechestError)
    message = str(err)
    assert "granules" in message
    assert "granule_id == 'g1'" in message
    assert "11" in message and "22" in message


def test_unreplayable_change_error_names_the_tables():
    err = UnreplayableChangeError(["granules", "quality"])

    assert err.tables == ["granules", "quality"]
    assert isinstance(err, IcechestError)
    assert "tx.delete()" in str(err)
