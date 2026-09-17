# Icechunk–Iceberg Convention

- **Identifier (UUID):** `66da9667-d40f-46cb-93a5-7c915af5f7dc`
- **Namespace prefix:** `iceberg:`
- **Version:** 1
- **Status:** experimental

A Zarr convention that binds a Zarr group to one or more [Apache Iceberg](https://iceberg.apache.org/)
tables describing the arrays in that group, with the current table version
resolved from the enclosing [Icechunk](https://icechunk.io/) snapshot rather
than from an Iceberg catalog service.

The keywords in this document follow [RFC2119](https://www.rfc-editor.org/rfc/rfc2119).

## Motivation

Large array collections need a queryable index: which granules exist, over what
time range, in which projection, at which array path. Today that index lives in
a separate system (STAC, CMR, ODC) that references data in object storage, and
keeping the two consistent requires fragile orchestration.

Iceberg is a good fit for the index — a mature columnar table format with broad
engine support. But Iceberg resolves the current table version through a catalog
service, which is a second source of truth that can disagree with the arrays.

This convention removes the catalog from that path. The Icechunk snapshot both
holds the arrays and names the table version describing them, so a single commit
advances the two together and an Icechunk tag pins both.

## Scope

This convention is specific to Zarr stores backed by Icechunk. It relies on
Icechunk commit metadata to carry the table version, and has no meaning for a
Zarr store without it.

## Convention Properties

A node declaring this convention MUST include the Convention Metadata Object in
its `zarr_conventions` attribute, and MUST include the following attributes.

### `iceberg:version`

**Type:** string. **Required.**

The version of this convention. MUST be `"1"`.

### `iceberg:tables`

**Type:** object. **Required.**

A mapping of table name to *table binding*. Table names MUST be unique within a
node. A binding has one field:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `location` | string | yes | Root URI of the Iceberg table's files (data, manifests, metadata). |

The binding is written when the table is created and MUST NOT change as rows are
appended. This is what keeps concurrent writers viable — see below.

## Resolving the current table version

The current `metadata.json` for each declared table is recorded in the **Icechunk
commit metadata** of the snapshot being read, under the key
`iceberg:table_versions`, as a mapping of table name to URI:

```json
{
  "iceberg:table_versions": {
    "granules": "s3://veda-odd/warehouse/granules/metadata/00042-1f0c9d3e.metadata.json"
  }
}
```

The two attributes answer different questions. To learn which tables a
snapshot holds, a reader reads `iceberg:tables` from the group attributes. To
read a table it can already name, a reader:

1. reads `iceberg:table_versions` from that snapshot's commit metadata;
2. loads the `metadata.json` at that table's URI.

Resolution does not go through the group. The pointer is per-snapshot and
arrives with the snapshot already in hand, so a reader holding a table's name
needs nothing else; discovery is what the declaration is for.

Writers MUST write a table's declaration and its first pointer in the same
commit, and MUST include an entry for every declared table in every subsequent
commit's `iceberg:table_versions`, carrying unchanged tables forward. Together
these mean **a snapshot that declares a table always resolves it**: a snapshot
predating a table declares neither half, and no snapshot sees one half without
the other.

A declared table missing from `iceberg:table_versions` therefore indicates a
malformed store. Readers MUST report it rather than falling back to a version
from another snapshot, and SHOULD continue to expose the arrays. A reader
resolving by name meets this as a missing pointer, and reads the declaration
then -- to tell a malformed store from a table this snapshot simply does not
have.

## Why the version is not in the attributes

The obvious design puts the current `metadata.json` URI directly in the group
attributes. It is rejected here for a concrete reason.

The pointer changes on *every* table commit, so every writer would rewrite the
same Zarr node. In Icechunk that surfaces as a `ZarrMetadataDoubleUpdate`
conflict on `/`, which `BasicConflictSolver` has no strategy for — two writers
appending to disjoint array regions could not rebase, even though their array
writes do not conflict.

Keeping the stable declaration in the attributes and the volatile version in
commit metadata removes the contention entirely: attributes are written once at
table creation, and commit metadata is per-snapshot and set at commit time.

## Garbage collection

Consumers MUST NOT run Iceberg's own expiry or orphan-file cleanup against these
tables. Those tools judge deletability from the table's current version alone and
will delete files that an older Icechunk tag still pins.

Tables SHOULD therefore be created with:

```
gc.enabled = false
write.metadata.delete-after-commit.enabled = false
```

Because table properties only constrain well-behaved clients, deployments SHOULD
also deny delete permissions on the warehouse prefix at the storage layer.
Reclaiming space requires a sweeper that treats Icechunk refs as the source of
truth: walk every live branch and tag, collect the pinned `metadata.json` from
each snapshot's commit metadata, follow each to the files it references, and
delete only what nothing reaches.

## Relationship to other conventions

This convention describes *where the table is*, not what is in it; the table's
schema is Iceberg's concern. Conventions such as `stac` may describe the same
collection from the Zarr side. They compose, as convention properties all live at
the root `attributes` level.

## Examples

See [`examples/`](examples/): a minimal binding, and one composed with another
convention.
