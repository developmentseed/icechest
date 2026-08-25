"""A PyIceberg catalog whose source of truth is an Icechunk snapshot.

``IcechunkCatalog`` implements just enough of PyIceberg's ``Catalog`` interface
to create tables, load them, and commit new versions -- with no catalog service
anywhere. ``load_table`` resolves the current ``metadata.json`` from a pointer
map handed in by the caller, and ``commit_table`` writes the new
``metadata.json`` to object storage and *stages* its location rather than
publishing it.

Staging is what makes the design atomic: nothing is visible to readers until an
Icechunk commit carries the staged pointer into a snapshot. Until then a
half-finished write is just unreferenced files in the warehouse.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from pyiceberg.catalog import Catalog, PropertiesUpdateSummary
from pyiceberg.exceptions import NoSuchTableError, TableAlreadyExistsError
from pyiceberg.io import FileIO, load_file_io
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.serializers import FromInputFile, ToOutputFile
from pyiceberg.table import CommitTableResponse, Table
from pyiceberg.table.metadata import new_table_metadata
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER, SortOrder
from pyiceberg.table.update import update_table_metadata

if TYPE_CHECKING:
    from pyiceberg.table.update import TableRequirement, TableUpdate
    from pyiceberg.typedef import Identifier, Properties

#: Table properties that keep Iceberg's own cleanup machinery switched off.
#:
#: Icechunk tags can pin a ``metadata.json`` from months ago, and every file
#: that version references must stay on disk. Iceberg's expiry and orphan-file
#: tooling judges deletability from the table's *current* version alone, so it
#: would happily delete files a tagged Icechunk snapshot still needs. See the
#: garbage-collection discussion in the README.
SAFETY_PROPERTIES: dict[str, str] = {
    "gc.enabled": "false",
    "write.metadata.delete-after-commit.enabled": "false",
}


class IcechunkCatalog(Catalog):
    """Resolve and commit Iceberg tables against an Icechunk-held pointer.

    Parameters
    ----------
    warehouse:
        Root location for Iceberg table files (data, manifests, metadata).
    pointers:
        The ``{table_name: metadata_location}`` map as of the session's base
        snapshot. Read-only; staged changes go to :attr:`staged`.
    properties:
        FileIO properties (e.g. S3 credentials).
    """

    def __init__(
        self,
        name: str = "icechest",
        *,
        warehouse: str,
        pointers: dict[str, str] | None = None,
        properties: Properties | None = None,
    ) -> None:
        super().__init__(name, **(properties or {}))
        self.warehouse = warehouse.rstrip("/")
        self.pointers: dict[str, str] = dict(pointers or {})
        #: Pointer updates produced this session, not yet published by Icechunk.
        self.staged: dict[str, str] = {}
        self._io: FileIO = load_file_io(properties=dict(self.properties))

    # -- pointer plumbing ---------------------------------------------------

    def current_pointers(self) -> dict[str, str]:
        """Base pointers overlaid with anything staged this session."""
        return {**self.pointers, **self.staged}

    def rebase_onto(self, pointers: dict[str, str]) -> None:
        """Adopt a new base pointer map, discarding staged locations.

        Used by the retry loop after another writer commits: the staged
        ``metadata.json`` files were built on a stale parent and are abandoned
        (they become unreferenced files, swept later by garbage collection).
        """
        self.pointers = dict(pointers)
        self.staged.clear()

    def _metadata_location(self, table_name: str, version: int) -> str:
        token = uuid.uuid4()
        return (
            f"{self.warehouse}/{table_name}/metadata/"
            f"{version:05d}-{token}.metadata.json"
        )

    def _resolve(self, identifier: str | Identifier) -> str:
        name = self.table_name_from(identifier)
        pointers = self.current_pointers()
        if name not in pointers:
            raise NoSuchTableError(f"Table not in Icechunk pointer map: {name}")
        return pointers[name]

    # -- Catalog interface --------------------------------------------------

    def create_table(
        self,
        identifier: str | Identifier,
        schema: Schema | Any,
        location: str | None = None,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
        sort_order: SortOrder = UNSORTED_SORT_ORDER,
        properties: Properties = {},  # noqa: B006 - matches PyIceberg's signature
    ) -> Table:
        name = self.table_name_from(identifier)
        if name in self.current_pointers():
            raise TableAlreadyExistsError(f"Table already exists: {name}")

        schema = self._convert_schema_if_needed(schema)
        location = (location or f"{self.warehouse}/{name}").rstrip("/")

        metadata = new_table_metadata(
            schema=schema,
            partition_spec=partition_spec,
            sort_order=sort_order,
            location=location,
            properties={**SAFETY_PROPERTIES, **properties},
        )
        metadata_location = self._metadata_location(name, 0)
        ToOutputFile.table_metadata(metadata, self._io.new_output(metadata_location))
        self.staged[name] = metadata_location

        return Table(
            identifier=(name,),
            metadata=metadata,
            metadata_location=metadata_location,
            io=self._io,
            catalog=self,
        )

    def load_table(self, identifier: str | Identifier) -> Table:
        name = self.table_name_from(identifier)
        metadata_location = self._resolve(identifier)
        metadata = FromInputFile.table_metadata(self._io.new_input(metadata_location))
        return Table(
            identifier=(name,),
            metadata=metadata,
            metadata_location=metadata_location,
            io=self._io,
            catalog=self,
        )

    def commit_table(
        self,
        table: Table,
        requirements: tuple[TableRequirement, ...],
        updates: tuple[TableUpdate, ...],
    ) -> CommitTableResponse:
        """Write a new ``metadata.json`` and stage its location.

        Note what is *not* here: no compare-and-swap against a catalog. The
        atomicity check happens later, when Icechunk commits the staged pointer
        and detects that another writer moved the branch tip.
        """
        name = self.table_name_from(table.name())
        base_location = self.current_pointers().get(name)
        base_metadata = (
            FromInputFile.table_metadata(self._io.new_input(base_location))
            if base_location
            else table.metadata
        )

        for requirement in requirements:
            requirement.validate(base_metadata)

        new_metadata = update_table_metadata(
            base_metadata,
            updates,
            enforce_validation=False,
            metadata_location=base_location,
        )
        version = len(new_metadata.metadata_log)
        metadata_location = self._metadata_location(name, version)
        ToOutputFile.table_metadata(
            new_metadata, self._io.new_output(metadata_location)
        )
        self.staged[name] = metadata_location

        return CommitTableResponse(
            metadata=new_metadata, metadata_location=metadata_location
        )

    def table_exists(self, identifier: str | Identifier) -> bool:
        return self.table_name_from(identifier) in self.current_pointers()

    def list_tables(self, namespace: str | Identifier = ()) -> list[Identifier]:
        return [(name,) for name in sorted(self.current_pointers())]

    # -- deliberately unsupported -------------------------------------------
    #
    # Dropping or renaming a table would orphan files that older Icechunk tags
    # still reference. Namespaces and views have no meaning without a catalog
    # service. All are refused rather than silently no-op'd.

    def _unsupported(self, what: str) -> NotImplementedError:
        return NotImplementedError(
            f"{what} is not supported by IcechunkCatalog: the Icechunk snapshot "
            "is the source of truth, and this operation would strand files that "
            "existing tags still reference."
        )

    def drop_table(self, identifier: str | Identifier) -> None:
        raise self._unsupported("drop_table")

    def purge_table(self, identifier: str | Identifier) -> None:
        raise self._unsupported("purge_table")

    def rename_table(
        self, from_identifier: str | Identifier, to_identifier: str | Identifier
    ) -> Table:
        raise self._unsupported("rename_table")

    def register_table(
        self, identifier: str | Identifier, metadata_location: str
    ) -> Table:
        raise self._unsupported("register_table")

    def create_namespace(
        self,
        namespace: str | Identifier,
        properties: Properties = {},  # noqa: B006
    ) -> None:
        raise self._unsupported("create_namespace")

    def drop_namespace(self, namespace: str | Identifier) -> None:
        raise self._unsupported("drop_namespace")

    def list_namespaces(self, namespace: str | Identifier = ()) -> list[Identifier]:
        return []

    def namespace_exists(self, namespace: str | Identifier) -> bool:
        return False

    def load_namespace_properties(self, namespace: str | Identifier) -> Properties:
        raise self._unsupported("load_namespace_properties")

    def update_namespace_properties(
        self,
        namespace: str | Identifier,
        removals: set[str] | None = None,
        updates: Properties = {},  # noqa: B006
    ) -> PropertiesUpdateSummary:
        raise self._unsupported("update_namespace_properties")

    def create_table_transaction(self, *args: Any, **kwargs: Any) -> Any:
        raise self._unsupported("create_table_transaction")

    def drop_view(self, identifier: str | Identifier) -> None:
        raise self._unsupported("drop_view")

    def list_views(self, namespace: str | Identifier) -> list[Identifier]:
        return []

    def view_exists(self, identifier: str | Identifier) -> bool:
        return False
