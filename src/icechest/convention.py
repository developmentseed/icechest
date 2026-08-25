"""The ``iceberg:`` Zarr convention -- how a Zarr group declares its tables.

A Zarr Convention is a set of attributes that confer special meaning on a node
and are *safely ignorable* by low-level Zarr implementations. That is the right
shape for the binding between array data and the Iceberg table describing it: a
reader that knows nothing about Iceberg still sees a valid Zarr hierarchy.

The attributes declare which tables exist and where their files live. They do
not carry the current ``metadata.json`` location -- that is resolved from the
Icechunk snapshot being read (see :mod:`icechest.pointer`), which is what makes
arrays and table version advance together.

Keeping the live pointer out of the attributes is also what makes concurrent
writers possible at all. A pointer in the attributes would be rewritten by every
writer on every commit; Icechunk reports that as a ``ZarrMetadataDoubleUpdate``
conflict on ``/`` that ``BasicConflictSolver`` cannot resolve, so two writers
appending to disjoint array regions would be unable to rebase. Declarations
change only when a table is created, so appends never touch the Zarr node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import zarr

CONVENTION_UUID = "66da9667-d40f-46cb-93a5-7c915af5f7dc"
CONVENTION_NAME = "iceberg:"
CONVENTION_VERSION = "1"
SCHEMA_URL = (
    "https://raw.githubusercontent.com/developmentseed/icechest/"
    "refs/tags/v1/conventions/icechunk-iceberg/schema.json"
)
SPEC_URL = (
    "https://github.com/developmentseed/icechest/blob/v1/"
    "conventions/icechunk-iceberg/README.md"
)

CONVENTIONS_KEY = "zarr_conventions"
TABLES_ATTR = "iceberg:tables"
VERSION_ATTR = "iceberg:version"

CONVENTION_METADATA_OBJECT: dict[str, str] = {
    "uuid": CONVENTION_UUID,
    "schema_url": SCHEMA_URL,
    "spec_url": SPEC_URL,
    "name": CONVENTION_NAME,
    "description": (
        "Binds a Zarr group to Apache Iceberg tables describing its arrays, "
        "with the current table version resolved from the Icechunk snapshot."
    ),
}


@dataclass(frozen=True)
class TableBinding:
    """One Zarr group -> Iceberg table binding, as declared in the attributes."""

    location: str

    def to_attrs(self) -> dict[str, Any]:
        return {"location": self.location}

    @classmethod
    def from_attrs(cls, raw: dict[str, Any]) -> TableBinding:
        return cls(location=raw["location"])


def declare(group: zarr.Group, bindings: dict[str, TableBinding]) -> None:
    """Register the convention on ``group`` and merge in table bindings.

    Idempotent, and writes only when something actually changes -- re-declaring
    an unchanged binding must not dirty the Zarr node, or it would reintroduce
    the attribute conflict this design avoids.
    """
    attrs = dict(group.attrs)

    declared = list(attrs.get(CONVENTIONS_KEY) or [])
    if not any(cmo.get("uuid") == CONVENTION_UUID for cmo in declared):
        declared.append(CONVENTION_METADATA_OBJECT)

    tables = dict(attrs.get(TABLES_ATTR) or {})
    tables.update({name: b.to_attrs() for name, b in bindings.items()})

    updated = {
        CONVENTIONS_KEY: declared,
        VERSION_ATTR: CONVENTION_VERSION,
        TABLES_ATTR: tables,
    }
    if all(attrs.get(k) == v for k, v in updated.items()):
        return
    group.attrs.update(updated)


def read_bindings(group: zarr.Group) -> dict[str, TableBinding]:
    """Return the table bindings declared on ``group``, empty if none."""
    raw = dict(group.attrs).get(TABLES_ATTR) or {}
    return {name: TableBinding.from_attrs(b) for name, b in raw.items()}


def is_declared(group: zarr.Group) -> bool:
    """Whether ``group`` declares this convention."""
    declared = dict(group.attrs).get(CONVENTIONS_KEY) or []
    return any(cmo.get("uuid") == CONVENTION_UUID for cmo in declared)
