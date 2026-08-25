"""Atomic array + table commits, and the retry loop that keeps them atomic.

A :class:`HybridTransaction` collects two kinds of work: Zarr array writes,
staged directly in an Icechunk session, and Iceberg table operations, recorded
as *intents* and applied at commit time.

Recording table work as intents rather than applying it eagerly is what makes
conflict recovery possible. When another writer lands first, the Iceberg
``metadata.json`` we built has the wrong parent, so it has to be rebuilt on top
of theirs. Replaying an intent does exactly that; there is nothing to unwind
because nothing was ever published.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any

import icechunk
import zarr

from icechest.catalog import IcechunkCatalog
from icechest.convention import TableBinding, declare, read_bindings
from icechest.pointer import (
    commit_metadata,
    read_pointers,
    read_pointers_at_branch,
    resolve_metadata_location,
)

if TYPE_CHECKING:
    import pyarrow as pa
    from pyiceberg.schema import Schema

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 10


class TableIntent:
    """A table operation to (re)apply against whatever the current parent is."""

    def apply(self, catalog: IcechunkCatalog) -> None:
        raise NotImplementedError


@dataclass
class CreateTable(TableIntent):
    name: str
    schema: Schema | pa.Schema
    location: str
    properties: dict[str, str] = field(default_factory=dict)

    def apply(self, catalog: IcechunkCatalog) -> None:
        if catalog.table_exists(self.name):
            return
        catalog.create_table(
            self.name,
            self.schema,
            location=self.location,
            properties=self.properties,
        )


@dataclass
class AppendRows(TableIntent):
    name: str
    data: pa.Table

    def apply(self, catalog: IcechunkCatalog) -> None:
        catalog.load_table(self.name).append(self.data)


class HybridTransaction:
    """Stage array writes and table operations, then publish them as one commit."""

    def __init__(
        self,
        repo: HybridRepo,
        branch: str,
        message: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._repo = repo
        self._branch = branch
        self._message = message
        self._max_retries = max_retries
        self._intents: list[TableIntent] = []

        self.session = repo.repo.writable_session(branch)
        self.catalog = repo._catalog_for(
            read_pointers(repo.repo, self.session.snapshot_id)
        )
        #: Snapshot ID produced by this transaction, set once committed.
        self.snapshot_id: str | None = None

    # -- staging ------------------------------------------------------------

    @property
    def group(self) -> zarr.Group:
        """The Zarr root group, writable through the Icechunk session."""
        return zarr.open_group(self.session.store, mode="a")

    def create_table(
        self,
        name: str,
        schema: Schema | pa.Schema,
        *,
        location: str | None = None,
        properties: dict[str, str] | None = None,
    ) -> None:
        """Create a table and declare it on the Zarr group.

        The binding goes into the group attributes under the ``iceberg:``
        convention, so the store is self-describing: a reader discovers the
        table from the Zarr hierarchy alone, with no catalog and no
        out-of-band configuration.
        """
        location = (location or f"{self._repo.warehouse}/{name}").rstrip("/")
        declare(self.group, {name: TableBinding(location=location)})
        self._intents.append(CreateTable(name, schema, location, properties or {}))

    def append(self, name: str, data: pa.Table) -> None:
        """Append rows to a table. Appends are what the retry loop can replay."""
        self._intents.append(AppendRows(name, data))

    # -- commit -------------------------------------------------------------

    def commit(self) -> str:
        """Publish arrays and table version in one Icechunk commit.

        Retries on conflict by rebuilding the Iceberg metadata on the winning
        writer's version and rebasing the staged array chunks onto their
        snapshot.
        """
        for attempt in range(self._max_retries + 1):
            for intent in self._intents:
                intent.apply(self.catalog)

            pointers = self.catalog.current_pointers()
            # A table-only transaction changes no Zarr nodes, so Icechunk sees
            # an empty commit -- but the pointer in the commit metadata *did*
            # move, which is a real change in this design. Allow it explicitly.
            table_only = (
                bool(self._intents) and not self.session.has_uncommitted_changes
            )
            try:
                self.snapshot_id = self.session.commit(
                    self._message,
                    metadata=commit_metadata(pointers),
                    allow_empty=table_only,
                )
            except icechunk.ConflictError:
                if attempt == self._max_retries:
                    raise
                logger.info(
                    "commit conflict on %r (attempt %d); replaying onto new tip",
                    self._branch,
                    attempt + 1,
                )
                self._recover()
                continue
            return self.snapshot_id

        raise AssertionError("unreachable")  # pragma: no cover

    def _recover(self) -> None:
        """Adopt the winning writer's table version and rebase our array writes.

        Order matters. The pointer map is re-read *before* the rebase so that
        the replayed Iceberg metadata descends from the version that actually
        won, and the rebase then carries our staged chunks forward onto their
        snapshot, re-checking them for genuine array-level conflicts.
        """
        tip_pointers = read_pointers_at_branch(self._repo.repo, self._branch)
        self.catalog.rebase_onto(tip_pointers)
        self.session.rebase(icechunk.BasicConflictSolver())

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> HybridTransaction:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.session.discard_changes()
            return
        self.commit()


class HybridRepo:
    """An Icechunk repository plus the Iceberg warehouse it points at."""

    def __init__(
        self,
        repo: icechunk.Repository,
        warehouse: str,
        *,
        io_properties: dict[str, str] | None = None,
    ) -> None:
        self.repo = repo
        self.warehouse = warehouse.rstrip("/")
        self.io_properties = dict(io_properties or {})

    @classmethod
    def create(
        cls,
        storage: icechunk.Storage,
        warehouse: str,
        *,
        io_properties: dict[str, str] | None = None,
    ) -> HybridRepo:
        return cls(
            icechunk.Repository.open_or_create(storage),
            warehouse,
            io_properties=io_properties,
        )

    def _catalog_for(self, pointers: dict[str, str]) -> IcechunkCatalog:
        return IcechunkCatalog(
            warehouse=self.warehouse,
            pointers=pointers,
            properties=self.io_properties,
        )

    def transaction(
        self,
        branch: str = "main",
        message: str = "",
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> HybridTransaction:
        return HybridTransaction(self, branch, message, max_retries=max_retries)

    # -- reading ------------------------------------------------------------

    def _snapshot_for(
        self, branch: str | None, tag: str | None, snapshot_id: str | None
    ) -> str:
        given = [x for x in (branch, tag, snapshot_id) if x is not None]
        if len(given) != 1:
            raise ValueError("pass exactly one of branch, tag, or snapshot_id")
        if snapshot_id is not None:
            return snapshot_id
        if tag is not None:
            return self.repo.lookup_tag(tag)
        return self.repo.lookup_branch(branch)  # type: ignore[arg-type]

    def read(
        self,
        branch: str | None = None,
        *,
        tag: str | None = None,
        snapshot_id: str | None = None,
    ) -> HybridSnapshot:
        """Open a consistent read view of arrays and tables at one snapshot."""
        if branch is None and tag is None and snapshot_id is None:
            branch = "main"
        return HybridSnapshot(self, self._snapshot_for(branch, tag, snapshot_id))


@dataclass
class HybridSnapshot:
    """Arrays and the table version that describes them, at a single snapshot."""

    _repo: HybridRepo
    snapshot_id: str

    @property
    def group(self) -> zarr.Group:
        session = self._repo.repo.readonly_session(snapshot_id=self.snapshot_id)
        return zarr.open_group(session.store, mode="r")

    @property
    def pointers(self) -> dict[str, str]:
        return read_pointers(self._repo.repo, self.snapshot_id)

    @property
    def bindings(self) -> dict[str, TableBinding]:
        """Table bindings declared by the ``iceberg:`` convention."""
        return read_bindings(self.group)

    def table(self, name: str) -> Any:
        """Load a table, resolving its version the way the convention says to."""
        bindings = self.bindings
        if name not in bindings:
            raise KeyError(
                f"Table {name!r} is not declared on this Zarr group. "
                f"Declared: {sorted(bindings)}"
            )
        location = resolve_metadata_location(
            name, self._repo.repo, self.snapshot_id
        )
        catalog = self._repo._catalog_for({name: location})
        return catalog.load_table(name)
