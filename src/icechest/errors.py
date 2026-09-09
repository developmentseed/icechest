"""Errors raised when table work cannot be published.

Both of these mean the same thing to a caller: nothing was written, and the
transaction has to be re-planned against the current tip. They differ in why
the work could not be carried forward.
"""

from __future__ import annotations

from typing import Any


class IcechestError(Exception):
    """Base class for errors raised by icechest."""


class TableConflictError(IcechestError):
    """A table operation cannot be replayed onto the winning writer's version.

    Raised when another writer added rows matching the operation's predicate
    between the version we planned against and the version that won. Replaying
    would apply our intent to rows we never saw.
    """

    def __init__(self, table: str, predicate: Any, snapshot_ids: list[int]) -> None:
        self.table = table
        self.predicate = predicate
        self.snapshot_ids = list(snapshot_ids)
        super().__init__(
            f"Cannot replay the operation on table {table!r}: another writer "
            f"added rows matching {predicate!r} in snapshots {self.snapshot_ids}. "
            "Nothing was published; re-plan against the current tip and retry."
        )


class UnreplayableChangeError(IcechestError):
    """Staged table work that no intent can rebuild after a conflict.

    Recovery rebuilds the Iceberg metadata by replaying intents, so anything
    staged outside that record would be silently dropped. Refusing is the only
    safe response.
    """

    def __init__(self, tables: list[str]) -> None:
        self.tables = list(tables)
        super().__init__(
            f"Tables {self.tables} have staged changes this transaction cannot "
            "replay onto another writer's version, so recovering from the "
            "conflict would silently discard them. Record table work with "
            "tx.append(), tx.delete() or tx.overwrite() rather than calling the "
            "catalog directly."
        )
