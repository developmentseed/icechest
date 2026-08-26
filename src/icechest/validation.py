"""Snapshot-isolation conflict validation for replayed table operations.

An append can always be replayed onto another writer's table version: adding
rows commutes with adding other rows. A delete cannot. Whether it is safe to
rebuild a delete on top of the version that won depends on what that writer
did -- specifically, whether they added rows our predicate matches.

That question is answered from the table's own manifests, by walking the
snapshots between the version we planned against and the version that won.

This is the only module that touches PyIceberg's private validation helpers.
Nothing inside PyIceberg 0.11 calls them -- they are staged for a later release
-- so their behaviour is pinned by our tests rather than trusted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyiceberg.exceptions import ValidationException
from pyiceberg.expressions.parser import parse
from pyiceberg.table.update.validate import _added_data_files

if TYPE_CHECKING:
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.table import Table


def as_predicate(predicate: str | BooleanExpression) -> BooleanExpression:
    """Parse a row filter written as a string; pass expressions through."""
    return parse(predicate) if isinstance(predicate, str) else predicate


def conflicting_adds(
    table: Table,
    *,
    base_snapshot_id: int,
    tip_snapshot_id: int,
    predicate: str | BooleanExpression,
) -> list[int]:
    """Snapshot ids that added rows matching ``predicate`` after ``base``.

    ``table`` must be loaded at the winning writer's version, so that its
    metadata contains both ends of the window.

    Two undocumented behaviours of the underlying helper are handled here.
    Its ``starting_snapshot`` argument is the *newest* snapshot and
    ``parent_snapshot`` the *oldest*, the opposite of what its docstring says.
    And its walk is inclusive of the base, so the base snapshot's own additions
    -- usually the very rows a delete targets -- are filtered out: the window
    is (base, tip], not [base, tip].

    Detection is conservative. Candidate files are judged from column
    statistics, so a file whose range covers the predicate but holds no
    matching row counts as a conflict. Refusing a replay that would have been
    safe is the right direction to err.
    """
    if base_snapshot_id == tip_snapshot_id:
        return []

    base = table.metadata.snapshot_by_id(base_snapshot_id)
    tip = table.metadata.snapshot_by_id(tip_snapshot_id)
    if base is None or tip is None:
        # The winner's history holds only one end of our window, so nothing
        # can be proven about the interval between them.
        return [tip_snapshot_id]

    try:
        entries = list(
            _added_data_files(
                table=table,
                starting_snapshot=tip,
                data_filter=as_predicate(predicate),
                partition_set=None,
                parent_snapshot=base,
            )
        )
    except ValidationException:
        # The walk back from the tip never reached our base: the histories
        # diverged, so the replay cannot be shown to be safe.
        return [tip_snapshot_id]

    return sorted(
        {
            entry.snapshot_id
            for entry in entries
            if entry.snapshot_id is not None and entry.snapshot_id != base_snapshot_id
        }
    )
