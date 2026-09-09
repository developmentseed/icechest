# icechest

A hybrid store where **Icechunk** manages array data and **Apache Iceberg** manages
the tabular metadata describing it.

Prototype for [NASA-IMPACT/veda-odd#460](https://github.com/NASA-IMPACT/veda-odd/issues/460).

## The problem

Collections of heterogeneous arrays can't be managed as a single data cube and
instead need a queryable index. Today that index often lives in a
separate system (STAC, CMR, etc) pointing at data in object storage, and keeping
the two consistent takes fragile orchestration.

An earlier experiment, [zarr-datafusion-search](https://github.com/developmentseed/zarr-datafusion-search),
solved the consistency problem by encoding columnar metadata as 1-D Zarr arrays
inside the Icechunk store. That works, but it re-implements a lot of Parquet functionality by hand and needs a bespoke query engine.

This prototype relies on Iceberg to manage tabular information and Icechunk to
manages arrays so we take advantage of both ecosystems' strengths without reinventing the
wheel.

## How it works

Iceberg resolves "what is the current version of this table?" through a catalog
service. This has the same issues as the current "separate system" metadata
stores in the community.  Changes to array data and metadata can easily become
desynchronized without fragile orchestration.

With Icechest the Icechunk snapshot *is* the catalog. Every commit carries the location of
the Iceberg `metadata.json` describing the arrays in that same commit, so a
single Icechunk commit publishes the chunks and the table version together or
not at all — and an Icechunk tag pins both.

```python
with repo.transaction("main", "ingest granule G-100") as tx:
    tx.group["reflectance"][0:10] = chunk      # array data
    tx.append("granules", granule_row)          # table row
# one commit; both or neither
```

The parent Zarr group declares its tables through the
[Icechunk–Iceberg convention](conventions/icechunk-iceberg/), so the store is
self-describing — a reader discovers the table from the Zarr hierarchy alone,
with no catalog and no out-of-band configuration.

## Where the pointer lives, and why

The convention's attributes declare which tables exist and where their files
live. They deliberately do **not** carry the corresponding `metadata.json`
location. This information lives in Icechunk commit metadata, keyed `iceberg:table_versions`.


The architecture uses this split model to support concurrency.  If the live pointer lived in the group attributes, every writer would rewrite the same
Zarr node on every commit. Icechunk reports that as a `ZarrMetadataDoubleUpdate`
conflict on the group, which the `BasicConflictSolver` can't resolve. So two writers
appending to *disjoint* array regions would be unable to rebase, even though
their array writes didn't actually conflict.

Commit metadata doesn't have that contention.  It is per-snapshot and set at
commit time and can be set after any Iceberg conflict-driven replay
onto another writer's table version.

## Concurrent writing

Ben commits first. Then Anna's metadata commit conflicts so her Iceberg operation is replayed onto Ben's table version and her staged chunks are rebased onto his snapshot:

Recording table work as *intents* rather than applying it eagerly is what makes
this possible. Replaying an intent rebuilds the `metadata.json` on the winning
parent and nothing needs unwinding because nothing was ever published. The
abandoned metadata files become dereferenced garbage that can be collected and
compacted later.

Appends replay unconditionally: adding rows commutes with adding other rows.
Deletes and overwrites cannot, so they are replayed only when it is provably
safe. Before adopting the winner's version, the delete's predicate is checked
against every commit that landed since the operation was planned. If nobody
added rows matching it, the operation is rebuilt on their version; if somebody
did, the transaction raises `TableConflictError` and publishes nothing — arrays
included — so the caller can re-plan against what actually changed.

The check is snapshot isolation, and it is conservative: candidate files are
judged from column statistics, so a delete can be refused when a matching row
merely might exist. But two concurrent deletes that touch the same data file
can never both succeed regardless — the loser's copy-on-write rewrite would
have to land on a file that genuinely still contains the other writer's
target rows, which is the operationally significant case for a store
expecting bulk retraction jobs.

## Garbage collection

Iceberg's expiry and orphan-file cleanup judge if files can be deleted from the table's
*current* version alone. So they would delete files that older active Icechunk tag still
pins. Tables are therefore created with `gc.enabled=false` and
`write.metadata.delete-after-commit.enabled=false`, and `IcechunkCatalog` blocks
`drop_table`, `purge_table` and `rename_table` operations

Table properties only constrain well-behaved clients, so a real deployment should
also deny delete permissions on the warehouse prefix at the storage layer.
Reclaiming space needs a sweeper that treats Icechunk refs as the source of
truth.

Deletes enlarge the problem. They are copy-on-write, so a delete that loses a
race leaves behind the data files it rewrote as well as its abandoned
`metadata.json` and manifests. Nothing points at those files — a sweeper finds
them by listing the warehouse and subtracting what live refs reach.

## Layout

```
src/icechest/
  catalog.py       An Icechunk managed PyIceberg catalog with no catalog service
  convention.py    The Iceberg pointer Zarr convention usage
  errors.py        Errors raised when table work cannot be replayed or published
  pointer.py       The table pointer in Icechunk commit metadata
  transaction.py   Iceberg commit management. Handles intents, atomic commit and conflict replay
  validation.py    The snapshot-isolation conflict check for replayed deletes and overwrites
conventions/icechunk-iceberg/
  README.md        The Iceberg pointer convention specification
  schema.json      JSON schema; examples/ validated against it in tests
```

## Try it

```bash
uv sync
uv run pytest
uv run python examples/concurrent_writers.py
```

## Status and open questions

Todo:

- **Garbage collection.** The garbage-collection story is currently "disable Iceberg's
  cleanup"; the Icechunk-ref-aware collector that makes space reclaimable is
  unimplemented.
- **Upsert and merge-on-read deletes.** Deletes and overwrites replay under
  snapshot isolation; `upsert` is not exposed, and deletes are copy-on-write
  because PyIceberg does not write delete files.
