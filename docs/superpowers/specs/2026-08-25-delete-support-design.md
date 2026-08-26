# Delete and overwrite support with conflict validation

**Date:** 2026-08-25
**Status:** approved, not yet implemented

## Problem

`HybridTransaction` records table work as intents and replays them onto the
winning writer's table version when Icechunk reports a commit conflict. Only
`CreateTable` and `AppendRows` exist. Appends are commutative, so replaying one
onto another writer's version is always sound, and the design works.

Deletes are not commutative, and there is no delete intent. The README says a
delete "must fail and re-plan". It does not. Reaching through to the catalog —
the only way to express a delete today — produces silent data retention:

```
seeded: ['G-100', 'G-200', 'G-300']
anna staged pointer: 00002-52eba6e3-….metadata.json
ben  -> VX59MGH50Q
   [log] commit conflict on 'main' (attempt 1); replaying onto new tip
anna -> RX4X45MMJB (NO EXCEPTION)
final rows: ['G-100', 'G-200', 'G-300', 'G-400']
G-100 still present after a 'successful' delete: True
```

Anna deleted `G-100`, Ben concurrently appended `G-400`, Ben won the race.
Anna's `commit()` returned a snapshot id with no exception, her array write
landed, and her delete evaporated.

The mechanism is `IcechunkCatalog.rebase_onto`, whose second line is
`self.staged.clear()` (`src/icechest/catalog.py:91`). Clearing staged state is
correct when everything staged can be rebuilt by replaying an intent. A delete
applied directly to the catalog has no intent, so it is discarded and the
transaction commits without it. The recovery machinery working as designed is
precisely what makes the failure silent.

Two smaller holes sit next to it:

- A delete-only transaction cannot commit at all, conflict or not.
  `table_only = bool(self._intents) and not self.session.has_uncommitted_changes`
  keys off the intent list, which a direct catalog delete never populates, so
  `allow_empty` stays `False` and Icechunk rejects the commit:
  `SessionStateError: cannot commit, no changes made to the session`.
- `repo.read("main").table("granules").delete(...)` returns cleanly and changes
  nothing. `HybridSnapshot.table()` builds a throwaway catalog, so the new
  `metadata.json` is written to the warehouse and immediately orphaned. Because
  `SAFETY_PROPERTIES` sets `gc.enabled=false`, nothing will ever collect it.

## Goals

Support the three scenarios this store needs:

1. **Re-ingest a granule** — replace one granule's row with a corrected version.
2. **Retract or expire granules** — bulk-remove rows by range.
3. **Delete a row and its array region together** — where the table delete and
   the array write are causally linked and half of it landing is worse than
   neither.

Under **snapshot isolation**: a delete replays onto the winner's version unless
the winner added rows matching its predicate. A disjoint concurrent append —
Ben ingesting `G-400` while Anna retracts the 2024 granules — must replay
cleanly, because in an append-heavy ingest workload a rule that failed on any
concurrent commit would fail nearly every delete.

## Non-goals

- `upsert`. It is `overwrite` for matched rows plus `append` for the rest, and
  nothing needs it yet.
- Merge-on-read deletes. pyiceberg does not write them; `Transaction.delete`
  warns and falls back to copy-on-write.
- Serializable isolation as a selectable mode.
- Optimizing the validation scan with partition or column statistics.

## Background: what upstream provides

pyiceberg 0.11.1, the pinned version, gives us nothing automatic. Its only
concurrency mechanism is the `AssertRefSnapshotId` precondition checked by the
catalog, and `IcechunkCatalog` opts out of that — it validates requirements
against metadata re-read from its own pointer, so they always pass. The real
compare-and-swap is Icechunk's branch tip. There is no retry loop anywhere in
0.11.1; `commit.retry.num-retries` appears only in a docstring.

`pyiceberg/table/update/validate.py` ships the primitives we need —
`_validation_history`, `_added_data_files`, `_validate_added_data_files` — but
nothing in 0.11.1 imports them. They are staged for a later release, which means
they are also untested in the direction we intend to use them.

Two facts about those functions matter for this design.

**The argument names are inverted relative to their docstrings.**
`_validate_added_data_files(table, starting_snapshot, data_filter, parent_snapshot)`
documents `starting_snapshot` as "snapshot current at the start of the
operation", but `_validation_history` walks
`ancestors_between(from_snapshot=parent_snapshot, to_snapshot=starting_snapshot)`,
and `ancestors_between` walks *ancestors of* `to_snapshot` until it reaches
`from_snapshot` (`table/snapshots.py:423-431`). So `starting_snapshot` must be
the **new tip** and `parent_snapshot` the **old base**. Upstream's own tests
confirm it, calling `starting_snapshot=newest_snapshot,
parent_snapshot=oldest_snapshot` throughout (`tests/table/test_validate.py`).
Passing them the documented way raises
`ValidationException("No matching snapshot found")` rather than silently
validating nothing.

**Upstream has since built the whole thing.** pyiceberg `main` has real
`COMMIT_NUM_RETRIES` properties with a refresh-and-re-execute retry loop, and
`_SnapshotProducer._validate_concurrency()` wiring all five validators against a
`CommitWindow`. We cannot inherit the retry loop — it is driven by the catalog
raising `CommitFailedException`, and our conflict is not knowable until Icechunk
rejects the branch-tip CAS, long after `commit_table` returned. We can inherit
the validation. This design wraps it behind our own interface so that adopting
0.12 changes one file.

## Design

### Intent contract

`TableIntent` gains a table name and a validation hook. `apply()` is unchanged.

```python
class TableIntent:
    name: str

    def apply(self, catalog: IcechunkCatalog) -> None: ...

    def validate(self, catalog: IcechunkCatalog, base_snapshot_id: int | None) -> None:
        """Raise TableConflictError if replaying onto the current pointer is unsafe."""
```

The default `validate()` is a no-op. That covers `CreateTable` and
`AppendRows`, whose commutativity is what makes today's replay sound. Two new
intents override it:

- `DeleteRows(name, predicate)` — `apply` calls
  `catalog.load_table(name).delete(predicate)`
- `OverwriteRows(name, data, predicate)` — `apply` calls
  `catalog.load_table(name).overwrite(data, overwrite_filter=predicate)`

Both share one `validate()` implementation. pyiceberg's `overwrite()` is
literally `delete()` followed by an append (`table/__init__.py:594-654`), and
the append half is commutative, so the delete half carries the entire conflict
hazard. Update needs no separate machinery.

The public API is `tx.delete(name, predicate)` and
`tx.overwrite(name, data, predicate)`, matching pyiceberg's spelling so the
operations read the same here as against a normal catalog.

An overwrite can emit two Iceberg snapshots inside one pyiceberg transaction.
That is invisible here: only the final `metadata.json` is ever carried into
commit metadata, so no reader can observe the state between the delete and the
append. Nothing needs to be done to get this property; it falls out of the
pointer design.

### Validation

A new `src/icechest/validation.py` exposes one function with honest names:

```python
def conflicting_adds(
    table: Table,
    *,
    base_snapshot_id: int,
    tip_snapshot_id: int,
    predicate: str | BooleanExpression,
) -> list[int]:
    """Snapshot ids that added rows matching `predicate` between base and tip."""
```

It parses string predicates with `pyiceberg.expressions.parser.parse` and calls
`_added_data_files` — the generator, not the `_validate_` wrapper — with
`starting_snapshot` bound to the **tip** and `parent_snapshot` bound to the
**base**, collecting `entry.snapshot_id` from each conflicting manifest entry.
Using the generator keeps the conflicting ids as data; the `_validate_` wrapper
only puts them in an exception message. This inversion gets a dedicated test so
that a 0.12 upgrade correcting the names fails loudly instead of quietly
validating an empty range.

Validation is conservative. `_added_data_files` filters candidate files with
`_InclusiveMetricsEvaluator`, which judges from column statistics, so a file
whose stats range covers the predicate but which contains no matching rows
counts as a conflict. The error direction is right: we refuse a replay that
would have been safe, rather than performing one that is not.

### The validation window

Each table's window has a fixed left edge and a moving right edge.

- **Left edge** — the table's Iceberg snapshot id as of transaction start,
  captured lazily the first time a delete or overwrite intent is recorded, and
  cached on the transaction. Capturing it lazily keeps the append-only hot path
  from paying an extra metadata read; caching it on the transaction rather than
  reading `catalog.pointers` later matters because `rebase_onto` replaces that
  map.
- **Right edge** — the tip's current snapshot at each recovery attempt.

The left edge stays fixed across retries while the right edge advances. In a
three-writer pile-up, re-anchoring the left edge to the most recent winner would
skip the middle writer's commits entirely.

Two edge cases:

- **No base pointer and no tip pointer** — the table is created in this
  transaction. Nothing can have been added to it, so validation passes.
- **No base pointer but a tip pointer exists** — another writer created a table
  with the same name concurrently. `CreateTable.apply` returns early when the
  table exists, so replaying would apply our delete to *their* table. This is a
  conflict and raises.

### Why added-files is the only check needed

Java's serializable mode also validates that files we planned to rewrite were
not deleted underneath us. We do not need that, because we never carry a planned
file list across attempts: replay calls `table.delete(predicate)` fresh against
the winner's metadata, re-planning the copy-on-write rewrite from their current
files. Stale file references cannot leak from one attempt into the next.

The cost of that is real and worth stating: every failed attempt leaves behind
rewritten Parquet files as well as an abandoned `metadata.json`, and with
`gc.enabled=false` nothing collects them. This enlarges the sweeper's job
described in the garbage-collection section of the README beyond metadata-only
orphans.

### Replay flow

```python
def _recover(self) -> None:
    tip_pointers = read_pointers_at_branch(self._repo.repo, self._branch)
    self._validate_replayable(tip_pointers)   # raises TableConflictError
    self._assert_staged_is_covered()          # raises UnreplayableChangeError
    self.catalog.rebase_onto(tip_pointers)
    self.session.rebase(icechunk.BasicConflictSolver())
```

`_validate_replayable` runs each intent's `validate()` against the tip, per
table. Because it runs before `rebase_onto`, a failure aborts with nothing
published: the staged `metadata.json` files are abandoned, the array writes are
discarded, and the caller re-plans. That is the correct outcome for the
causally-linked scenario, where a row delete and the array rewrite it describes
must not come apart.

`_assert_staged_is_covered` closes the silent-loss hole. Any key in
`catalog.staged` that no intent can reproduce means the retry loop is about to
discard real work, so it raises `UnreplayableChangeError` pointing at
`tx.delete()`. This converts the `tx.catalog.load_table(...).delete(...)` escape
hatch from data loss into an error.

### Errors

Both new exceptions live in a new `src/icechest/errors.py` and are exported from
the package root.

- `TableConflictError` — carries table name, predicate, and the conflicting
  snapshot ids, so a caller can re-plan against what actually changed.
- `UnreplayableChangeError` — carries the table names whose staged pointers no
  intent covers.

### Collateral fixes

- `table_only` keys off `bool(self.catalog.staged)` rather than
  `bool(self._intents)`, so a delete-only transaction commits. This stays
  correct across retries, since replay repopulates `staged`.
- A transaction that stages nothing and touches no arrays raises a `ValueError`
  naming the transaction, instead of Icechunk's `SessionStateError: cannot
  commit, no changes made to the session`. This also covers a delete matching
  zero rows, which stages nothing at all.
- `HybridSnapshot.table()` gets a read-only catalog whose `commit_table` raises,
  so the read path cannot orphan a `metadata.json`.
- `HybridTransaction.__exit__` wraps `commit()` so a failed commit discards the
  session rather than leaving it live.

## Testing

Test-driven: the first test is the Anna/Ben scenario from the Problem section
inverted into an assertion, and it must fail against current `main` for the
right reason — a surviving `G-100` — before any fix lands.

| Test | Asserts |
| --- | --- |
| delete replays over disjoint append | Anna deletes `G-100`, Ben appends `G-400`; both land, rows are `{G-200, G-300, G-400}` |
| delete fails on matching add | Ben appends a row matching Anna's predicate; `TableConflictError`, Anna's array writes unpublished |
| overwrite re-ingest replays | `G-100` replaced while a disjoint writer wins the race |
| row plus array region is atomic | causally-linked delete; both land or neither |
| direct catalog mutation refused | today's silent-loss path raises `UnreplayableChangeError` |
| delete-only transaction commits | the `table_only` fix |
| zero-match delete | commits when other work exists; clear `ValueError` when it is the only work |
| validator orientation | pins the `starting_snapshot`/`parent_snapshot` inversion |
| two winners before success | window widens; both winners' commits are validated |
| concurrent create of same table | delete against a concurrently created table raises |
| read path is read-only | `repo.read(...).table(x).delete(...)` raises |

## Documentation and dependencies

- `pyproject.toml` pins `pyiceberg>=0.11,<0.12`, because the design depends on
  private functions whose argument orientation contradicts their docstrings.
- README: the "Concurrent writing" section currently says non-append operations
  "must fail and re-plan… I'll be working on supporting this in a more automatic
  way with subsequent PRs", and "Status and open questions" lists non-append
  conflict re-planning as unimplemented. Both change.
- README garbage-collection section gains the note that failed delete attempts
  orphan Parquet as well as metadata.
- The convention spec needs no change. It describes where the pointer lives, and
  nothing here moves it.
