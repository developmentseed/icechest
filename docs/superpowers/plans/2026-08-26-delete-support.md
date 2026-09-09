# Delete and Overwrite Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make deletes and overwrites first-class, replayable table operations that survive a concurrent commit when snapshot isolation proves it is safe, and fail loudly when it is not.

**Architecture:** Table work is recorded as intents and applied at commit time; when Icechunk rejects the branch-tip compare-and-swap, the intents are replayed onto the winner's table version. Appends replay unconditionally because adding rows commutes. Deletes gain a `validate()` hook that walks the winner's manifests between the version we planned against and the version that won, and refuses the replay if rows matching the predicate appeared in that window.

**Tech Stack:** Python 3.12+, PyIceberg 0.11.x, Icechunk 2.1+, PyArrow, Zarr 3, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-25-delete-support-design.md`

## Global Constraints

- Python `>=3.12`. Every module starts with `from __future__ import annotations`.
- PyIceberg pinned `>=0.11,<0.12`. The design depends on private functions whose argument orientation contradicts their docstrings, so a minor upgrade must be a deliberate act.
- No new runtime dependencies.
- PyIceberg private API (`pyiceberg.table.update.validate`) may be imported **only** from `src/icechest/validation.py`. Nothing else in the package imports a leading-underscore name from PyIceberg.
- ruff: line length 88, double quotes, lint rules `E, F, I, B, UP`. Every commit must leave `uv run ruff check src tests` clean.
- Tests run with `uv run pytest`. The full suite must pass at every commit.
- Docstrings explain *why*, matching the existing house style in `src/icechest/`. Do not narrate what the code plainly does.

---

### Task 1: Shared test fixtures and the failing delete test

Existing fixtures live inside `tests/test_hybrid.py` and the new test module needs them. Move them to shared modules first, then write the test that drives everything else.

**Files:**
- Create: `tests/helpers.py`
- Create: `tests/conftest.py`
- Modify: `tests/test_hybrid.py:1-44` (imports, `GRANULE_SCHEMA` at :12, `granules` at :21, the `repo` fixture at :34, `seed` at :40)
- Create: `tests/test_delete.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `tests.helpers.GRANULE_SCHEMA`, `tests.helpers.granules(*ids, day="2026-01-01") -> pa.Table`, `tests.helpers.seed(repo, *ids) -> str`, and a `repo` pytest fixture yielding a `HybridRepo` on `tmp_path`.

- [ ] **Step 1: Create the shared builders**

Create `tests/helpers.py`:

```python
"""Builders shared by the test modules."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from icechest import HybridRepo

GRANULE_SCHEMA = pa.schema(
    [
        pa.field("granule_id", pa.string(), nullable=False),
        pa.field("datetime", pa.timestamp("ms"), nullable=False),
        pa.field("array_path", pa.string(), nullable=False),
    ]
)


def granules(*ids: str, day: str = "2026-01-01") -> pa.Table:
    return pa.table(
        {
            "granule_id": list(ids),
            "datetime": pa.array(
                [np.datetime64(day, "ms")] * len(ids), pa.timestamp("ms")
            ),
            "array_path": [f"/data/{g}" for g in ids],
        },
        schema=GRANULE_SCHEMA,
    )


def seed(repo: HybridRepo, *ids: str) -> str:
    """Create the array and the granules table, optionally with starting rows."""
    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("data", shape=(100,), dtype="f4", chunks=(10,))
        tx.create_table("granules", GRANULE_SCHEMA)
        if ids:
            tx.append("granules", granules(*ids))
    return tx.snapshot_id
```

- [ ] **Step 2: Create the fixture module**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures."""

from __future__ import annotations

import icechunk
import pytest

from icechest import HybridRepo


@pytest.fixture
def repo(tmp_path):
    store = icechunk.local_filesystem_storage(str(tmp_path / "icechunk"))
    return HybridRepo.create(store, warehouse=str(tmp_path / "warehouse"))
```

- [ ] **Step 3: Point the existing tests at the shared modules**

In `tests/test_hybrid.py`, delete the `GRANULE_SCHEMA` definition, the `granules` function, the `repo` fixture, and the `seed` function, and replace the import block at the top of the file with exactly:

```python
"""End-to-end tests for the hybrid Icechunk/Iceberg store."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from icechest import read_pointers_at_tag
from tests.helpers import GRANULE_SCHEMA, granules, seed
```

`icechunk` and `HybridRepo` are no longer used in that file; ruff's `F401` will flag them if they are left behind.

- [ ] **Step 4: Verify the refactor changed no behaviour**

Run: `uv run pytest -q`
Expected: PASS. `tests/test_hybrid.py` still contributes its 7 tests and the whole suite is green — this step is a pure move, so any change in the passing count means something was dropped.

- [ ] **Step 5: Write the failing test**

Create `tests/test_delete.py`:

```python
"""Deletes and overwrites: replay when provably safe, fail when not."""

from __future__ import annotations

import numpy as np

from tests.helpers import granules, seed


def test_delete_replays_over_disjoint_append(repo):
    """Anna retracts a granule while Ben ingests an unrelated one."""
    seed(repo, "g1", "g2", "g3")

    anna = repo.transaction("main", "anna retracts g1")
    ben = repo.transaction("main", "ben ingests g4")

    anna.group["data"][50:60] = np.full(10, 2.0, "f4")
    anna.delete("granules", "granule_id == 'g1'")

    ben.group["data"][0:10] = np.ones(10, "f4")
    ben.append("granules", granules("g4"))

    ben.commit()
    anna.commit()  # conflicts, validates, replays onto Ben's version

    snap = repo.read("main")
    rows = snap.table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g2", "g3", "g4"}
    assert snap.group["data"][0] == 1.0
    assert snap.group["data"][50] == 2.0
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `uv run pytest tests/test_delete.py -q`
Expected: FAIL with `AttributeError: 'HybridTransaction' object has no attribute 'delete'`

- [ ] **Step 7: Commit**

```bash
git add tests/helpers.py tests/conftest.py tests/test_hybrid.py tests/test_delete.py
git commit -m "test: share fixtures and add failing delete replay test"
```

---

### Task 2: Error types

**Files:**
- Create: `src/icechest/errors.py`
- Modify: `src/icechest/__init__.py:1-22` (imports and `__all__`)
- Create: `tests/test_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `icechest.errors.IcechestError`, `icechest.errors.TableConflictError(table: str, predicate: Any, snapshot_ids: list[int])` with attributes `.table`, `.predicate`, `.snapshot_ids`, and `icechest.errors.UnreplayableChangeError(tables: list[str])` with attribute `.tables`. Both re-exported from the package root.

- [ ] **Step 1: Write the failing test**

Create `tests/test_errors.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_errors.py -q`
Expected: FAIL with `ImportError: cannot import name 'IcechestError' from 'icechest'`

- [ ] **Step 3: Write the implementation**

Create `src/icechest/errors.py`:

```python
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
```

In `src/icechest/__init__.py`, add the import below the existing `from icechest.catalog import ...` line:

```python
from icechest.errors import (
    IcechestError,
    TableConflictError,
    UnreplayableChangeError,
)
```

and add `"IcechestError"`, `"TableConflictError"`, `"UnreplayableChangeError"` to `__all__`, keeping it alphabetically sorted as it already is.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_errors.py -q && uv run ruff check src tests`
Expected: PASS, no lint findings.

- [ ] **Step 5: Commit**

```bash
git add src/icechest/errors.py src/icechest/__init__.py tests/test_errors.py
git commit -m "feat: add TableConflictError and UnreplayableChangeError"
```

---

### Task 3: Conflict validation

The only module allowed to touch PyIceberg's private validation helpers. Two of their behaviours are load-bearing and neither is documented correctly upstream, so both are pinned by tests here.

**Files:**
- Create: `src/icechest/validation.py`
- Modify: `pyproject.toml:12` (the `pyiceberg>=0.11` dependency line)
- Create: `tests/test_validation.py`

**Interfaces:**
- Consumes: the `repo` fixture and `tests.helpers` from Task 1.
- Produces: `icechest.validation.conflicting_adds(table, *, base_snapshot_id: int, tip_snapshot_id: int, predicate: str | BooleanExpression) -> list[int]` and `icechest.validation.as_predicate(predicate) -> BooleanExpression`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validation.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_validation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'icechest.validation'`

- [ ] **Step 3: Write the implementation**

Create `src/icechest/validation.py`:

```python
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
```

In `pyproject.toml`, change the dependency line `"pyiceberg>=0.11",` to:

```toml
    # Pinned below 0.12: validation.py depends on private helpers whose
    # argument orientation contradicts their docstrings.
    "pyiceberg>=0.11,<0.12",
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_validation.py -q && uv run ruff check src tests`
Expected: PASS, no lint findings.

- [ ] **Step 5: Commit**

```bash
git add src/icechest/validation.py tests/test_validation.py pyproject.toml
git commit -m "feat: add snapshot-isolation conflict validation"
```

---

### Task 4: DeleteRows intent and replay validation

**Files:**
- Modify: `src/icechest/transaction.py:42-47` (`TableIntent`), `:68-75` (after `AppendRows`), `:76-105` (`HybridTransaction.__init__`), `:126-129` (after `append`), `:170-183` (`_recover`)
- Modify: `tests/test_delete.py`

**Interfaces:**
- Consumes: `icechest.errors.TableConflictError` (Task 2), `icechest.validation.conflicting_adds` (Task 3).
- Produces: `TableIntent.name: str` and `TableIntent.validate(catalog, base_snapshot_id) -> None`; `DeleteRows(name, predicate)`; `HybridTransaction.delete(name, predicate) -> None`; `HybridTransaction._base_snapshots: dict[str, int | None]`; `HybridTransaction._capture_base_snapshot(name) -> None`; `HybridTransaction._validate_replayable(tip_pointers) -> None`.

- [ ] **Step 1: Add the second failing test**

Append to `tests/test_delete.py`:

```python
def test_delete_fails_when_winner_adds_matching_row(repo):
    """Ben re-ingests the granule Anna is retracting: she must re-plan."""
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna retracts g1")
    ben = repo.transaction("main", "ben re-ingests g1")

    anna.group["data"][50:60] = np.full(10, 2.0, "f4")
    anna.delete("granules", "granule_id == 'g1'")

    ben.append("granules", granules("g1"))
    ben_snapshot = ben.commit()

    with pytest.raises(TableConflictError) as excinfo:
        anna.commit()

    assert excinfo.value.table == "granules"
    assert excinfo.value.snapshot_ids

    tip = repo.read("main")
    assert tip.snapshot_id == ben_snapshot  # Anna published nothing
    assert tip.group["data"][50] == 0.0  # her array write died with the delete
    assert tip.table("granules").scan().to_arrow().num_rows == 3
```

and extend the imports at the top of that file to:

```python
import numpy as np
import pytest

from icechest import TableConflictError
from tests.helpers import granules, seed
```

- [ ] **Step 2: Run both tests to verify they fail**

Run: `uv run pytest tests/test_delete.py -q`
Expected: FAIL — both tests error with `AttributeError: 'HybridTransaction' object has no attribute 'delete'`

- [ ] **Step 3: Extend the intent contract**

In `src/icechest/transaction.py`, replace the `TableIntent` class with:

```python
class TableIntent:
    """A table operation to (re)apply against whatever the current parent is."""

    name: str

    def apply(self, catalog: IcechunkCatalog) -> None:
        raise NotImplementedError

    def validate(self, catalog: IcechunkCatalog, base_snapshot_id: int | None) -> None:
        """Check that replaying onto ``catalog``'s version is safe.

        Adding rows commutes with adding other rows, so the default is a no-op.
        Operations that remove rows depend on what the other writer did and
        override this.
        """
```

- [ ] **Step 4: Add the delete intent and its validation**

In `src/icechest/transaction.py`, add these imports alongside the existing ones:

```python
from icechest.errors import TableConflictError
from icechest.validation import conflicting_adds
```

and under `TYPE_CHECKING`, add:

```python
    from pyiceberg.expressions import BooleanExpression
```

Then add below `AppendRows`:

```python
def _validate_removal(
    catalog: IcechunkCatalog,
    name: str,
    predicate: str | BooleanExpression,
    base_snapshot_id: int | None,
) -> None:
    """Refuse to replay a row-removing operation onto an incompatible version.

    ``catalog`` is loaded at the winning writer's pointers. Shared by delete and
    overwrite: PyIceberg's overwrite is a delete followed by an append, and the
    append half commutes, so the delete half carries the whole hazard.
    """
    if not catalog.table_exists(name):
        return  # our own CreateTable intent will make it

    table = catalog.load_table(name)
    tip_snapshot_id = table.metadata.current_snapshot_id
    if tip_snapshot_id is None:
        return  # the winner's table holds no rows at all

    if base_snapshot_id is None:
        # No version to anchor a window on: either the table did not exist when
        # we started, or it held no rows, and the winner's does. Nothing can be
        # proven, so refuse rather than delete rows we never saw.
        raise TableConflictError(name, predicate, [tip_snapshot_id])

    ids = conflicting_adds(
        table,
        base_snapshot_id=base_snapshot_id,
        tip_snapshot_id=tip_snapshot_id,
        predicate=predicate,
    )
    if ids:
        raise TableConflictError(name, predicate, ids)


@dataclass
class DeleteRows(TableIntent):
    name: str
    predicate: str | BooleanExpression

    def apply(self, catalog: IcechunkCatalog) -> None:
        catalog.load_table(self.name).delete(self.predicate)

    def validate(self, catalog: IcechunkCatalog, base_snapshot_id: int | None) -> None:
        _validate_removal(catalog, self.name, self.predicate, base_snapshot_id)
```

- [ ] **Step 5: Record base snapshots and expose `delete`**

In `HybridTransaction.__init__`, add below `self._intents: list[TableIntent] = []`:

```python
        #: Each table's Iceberg snapshot as of transaction start, captured the
        #: first time an operation needs a validation window.
        self._base_snapshots: dict[str, int | None] = {}
```

Add these methods after `append`:

```python
    def delete(self, name: str, predicate: str | BooleanExpression) -> None:
        """Delete rows matching ``predicate``.

        Replayable only when another writer's commits provably did not touch
        the rows in question; see :func:`_validate_removal`.
        """
        self._capture_base_snapshot(name)
        self._intents.append(DeleteRows(name, predicate))

    def _capture_base_snapshot(self, name: str) -> None:
        """Pin the left edge of this table's validation window.

        Captured lazily, because only row-removing operations need it and it
        costs a metadata read. Captured once, so retries validate the whole
        range back to where the operation was planned rather than re-anchoring
        on each new winner.
        """
        if name in self._base_snapshots:
            return
        if name not in self.catalog.pointers:
            self._base_snapshots[name] = None  # created in this transaction
            return
        metadata = self.catalog.load_table(name).metadata
        self._base_snapshots[name] = metadata.current_snapshot_id
```

- [ ] **Step 6: Validate before recovering**

Replace `HybridTransaction._recover` with:

```python
    def _recover(self) -> None:
        """Adopt the winning writer's table version and rebase our array writes.

        Order matters. Validation runs first, against the tip, so a refusal
        aborts with nothing published. The pointer map is then adopted *before*
        the rebase so the replayed Iceberg metadata descends from the version
        that actually won, and the rebase carries our staged chunks forward
        onto their snapshot, re-checking them for genuine array-level
        conflicts.
        """
        tip_pointers = read_pointers_at_branch(self._repo.repo, self._branch)
        self._validate_replayable(tip_pointers)
        self.catalog.rebase_onto(tip_pointers)
        self.session.rebase(icechunk.BasicConflictSolver())

    def _validate_replayable(self, tip_pointers: dict[str, str]) -> None:
        """Ask each intent whether it can be rebuilt on the winner's version."""
        tip_catalog = self._repo._catalog_for(tip_pointers)
        for intent in self._intents:
            intent.validate(tip_catalog, self._base_snapshots.get(intent.name))
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: PASS, no lint findings.

- [ ] **Step 8: Commit**

```bash
git add src/icechest/transaction.py tests/test_delete.py
git commit -m "feat: add replayable delete with snapshot-isolation validation"
```

---

### Task 5: OverwriteRows intent

**Files:**
- Modify: `src/icechest/transaction.py` (after `DeleteRows`, and after `delete`)
- Modify: `tests/test_delete.py`

**Interfaces:**
- Consumes: `_validate_removal` and `HybridTransaction._capture_base_snapshot` (Task 4).
- Produces: `OverwriteRows(name, data, predicate)`; `HybridTransaction.overwrite(name, data, predicate) -> None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_delete.py`:

```python
def test_overwrite_reingest_replays_over_disjoint_append(repo):
    """Anna corrects g1's row while Ben ingests g9."""
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna re-ingests g1")
    ben = repo.transaction("main", "ben ingests g9")

    anna.overwrite(
        "granules", granules("g1", day="2026-02-02"), "granule_id == 'g1'"
    )
    anna.group["data"][50:60] = np.full(10, 2.0, "f4")

    ben.append("granules", granules("g9"))
    ben.commit()
    anna.commit()

    rows = repo.read("main").table("granules").scan().to_arrow()
    assert sorted(rows["granule_id"].to_pylist()) == ["g1", "g2", "g9"]
    corrected = dict(
        zip(rows["granule_id"].to_pylist(), rows["datetime"].to_pylist())
    )
    assert corrected["g1"].date().isoformat() == "2026-02-02"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_delete.py::test_overwrite_reingest_replays_over_disjoint_append -q`
Expected: FAIL with `AttributeError: 'HybridTransaction' object has no attribute 'overwrite'`

- [ ] **Step 3: Write the implementation**

In `src/icechest/transaction.py`, add below `DeleteRows`:

```python
@dataclass
class OverwriteRows(TableIntent):
    name: str
    data: pa.Table
    predicate: str | BooleanExpression

    def apply(self, catalog: IcechunkCatalog) -> None:
        catalog.load_table(self.name).overwrite(
            self.data, overwrite_filter=self.predicate
        )

    def validate(self, catalog: IcechunkCatalog, base_snapshot_id: int | None) -> None:
        _validate_removal(catalog, self.name, self.predicate, base_snapshot_id)
```

and add this method after `delete`:

```python
    def overwrite(
        self, name: str, data: pa.Table, predicate: str | BooleanExpression
    ) -> None:
        """Replace the rows matching ``predicate`` with ``data``.

        PyIceberg applies this as a delete followed by an append, which can
        produce two Iceberg snapshots. Readers never see the state between
        them: only the final ``metadata.json`` reaches commit metadata.
        """
        self._capture_base_snapshot(name)
        self._intents.append(OverwriteRows(name, data, predicate))
```

Note that `pa` is currently imported only under `TYPE_CHECKING`, which is all the annotations need since the module uses `from __future__ import annotations`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: PASS, no lint findings.

- [ ] **Step 5: Commit**

```bash
git add src/icechest/transaction.py tests/test_delete.py
git commit -m "feat: add replayable overwrite for granule re-ingest"
```

---

### Task 6: Refuse staged work no intent can replay

Recovery discards staged pointers on the assumption that replaying the intents rebuilds them. Anything staged outside that record is lost silently — the bug that motivated this work.

**Files:**
- Modify: `src/icechest/transaction.py` (`_recover`, plus a new method)
- Modify: `tests/test_delete.py`

**Interfaces:**
- Consumes: `icechest.errors.UnreplayableChangeError` (Task 2).
- Produces: `HybridTransaction._assert_staged_is_covered() -> None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_delete.py`:

```python
def test_direct_catalog_mutation_is_refused(repo):
    """The escape hatch that used to lose deletes silently now raises."""
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna deletes behind the intent API")
    ben = repo.transaction("main", "ben ingests g9")

    anna.group["data"][50:60] = np.full(10, 2.0, "f4")
    anna.catalog.load_table("granules").delete("granule_id == 'g1'")

    ben.append("granules", granules("g9"))
    ben.commit()

    with pytest.raises(UnreplayableChangeError, match="granules"):
        anna.commit()

    rows = repo.read("main").table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g1", "g2", "g9"}
```

and extend the import from `icechest` in that file to:

```python
from icechest import TableConflictError, UnreplayableChangeError
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_delete.py::test_direct_catalog_mutation_is_refused -q`
Expected: FAIL with `Failed: DID NOT RAISE <class 'icechest.errors.UnreplayableChangeError'>`. Anna's commit succeeds instead, and `g1` survives because the delete was discarded — the bug this task fixes.

- [ ] **Step 3: Write the implementation**

In `src/icechest/transaction.py`, add the import:

```python
from icechest.errors import TableConflictError, UnreplayableChangeError
```

(replacing the single-name import added in Task 4), add the call to `_recover` between validation and the rebase:

```python
        self._validate_replayable(tip_pointers)
        self._assert_staged_is_covered()
        self.catalog.rebase_onto(tip_pointers)
```

and add the method next to `_validate_replayable`:

```python
    def _assert_staged_is_covered(self) -> None:
        """Refuse to discard staged work that no intent can rebuild.

        ``rebase_onto`` abandons staged metadata because replaying the intents
        recreates it on the winning writer's version. A table touched outside
        the intent API breaks that assumption, and clearing it would drop the
        work with no error.
        """
        covered = {intent.name for intent in self._intents}
        orphaned = sorted(set(self.catalog.staged) - covered)
        if orphaned:
            raise UnreplayableChangeError(orphaned)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: PASS, no lint findings.

- [ ] **Step 5: Commit**

```bash
git add src/icechest/transaction.py tests/test_delete.py
git commit -m "fix: refuse staged table work that recovery cannot replay"
```

---

### Task 7: Commit bookkeeping

Three defects that only surface once a transaction's table work does not come from an append.

**Files:**
- Modify: `src/icechest/transaction.py:131-168` (`commit`), `:187-198` (`__exit__`)
- Create: `tests/test_commit_bookkeeping.py`

**Interfaces:**
- Consumes: `HybridTransaction.delete` (Task 4).
- Produces: no new names; `commit()` raises `ValueError` when a transaction changed nothing, and `__exit__` discards the session when `commit()` fails.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_commit_bookkeeping.py`:

```python
"""Commit-time bookkeeping for transactions whose table work is not an append."""

from __future__ import annotations

import numpy as np
import pytest

from icechest import TableConflictError
from tests.helpers import granules, seed


def test_delete_only_transaction_commits(repo):
    """No Zarr node changes, but the pointer moved: that is a real commit."""
    seed(repo, "g1", "g2")
    before = repo.read("main").snapshot_id

    with repo.transaction("main", "retract g1") as tx:
        tx.delete("granules", "granule_id == 'g1'")

    snap = repo.read("main")
    assert snap.snapshot_id != before
    rows = snap.table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g2"}


def test_transaction_with_no_changes_raises(repo):
    """A delete matching nothing stages nothing, so nothing was written."""
    seed(repo, "g1")

    with pytest.raises(ValueError, match="made no changes"):
        with repo.transaction("main", "zero-match retract") as tx:
            tx.delete("granules", "granule_id == 'no-such-granule'")


def test_zero_match_delete_commits_alongside_other_work(repo):
    seed(repo, "g1")

    with repo.transaction("main", "ingest and prune") as tx:
        tx.group["data"][0:10] = np.ones(10, "f4")
        tx.delete("granules", "granule_id == 'no-such-granule'")

    snap = repo.read("main")
    assert snap.group["data"][0] == 1.0
    assert snap.table("granules").scan().to_arrow().num_rows == 1


def test_failed_commit_discards_the_session(repo):
    """A refused replay must not leave array writes live on the session."""
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna retracts g1")
    ben = repo.transaction("main", "ben re-ingests g1")

    anna.group["data"][50:60] = np.full(10, 2.0, "f4")
    anna.delete("granules", "granule_id == 'g1'")

    ben.append("granules", granules("g1"))
    ben.commit()

    with pytest.raises(TableConflictError):
        with anna:
            pass

    assert anna.session.has_uncommitted_changes is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_commit_bookkeeping.py -q`
Expected: FAIL — `test_delete_only_transaction_commits` fails with `SessionStateError: cannot commit, no changes made to the session`, `test_transaction_with_no_changes_raises` fails with that same error instead of `ValueError`, and `test_failed_commit_discards_the_session` fails because the session still holds the array write.

- [ ] **Step 3: Fix the commit condition**

In `src/icechest/transaction.py`, replace the block inside `commit`'s retry loop that computes `pointers` and `table_only` with:

```python
            pointers = self.catalog.current_pointers()
            array_changes = self.session.has_uncommitted_changes
            if not self.catalog.staged and not array_changes:
                raise ValueError(
                    f"Transaction {self._message!r} made no changes: no array "
                    "writes, and no table operation moved a pointer. A delete "
                    "matching no rows stages nothing."
                )
            # A table-only transaction changes no Zarr nodes, so Icechunk sees
            # an empty commit -- but the pointer in the commit metadata *did*
            # move, which is a real change in this design. Allow it explicitly.
            # Keyed off staged pointers rather than the intent list, because an
            # intent can stage nothing and an append is not the only way to
            # move a pointer.
            table_only = bool(self.catalog.staged) and not array_changes
```

- [ ] **Step 4: Discard the session when the commit fails**

Replace `HybridTransaction.__exit__` with:

```python
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.session.discard_changes()
            return
        try:
            self.commit()
        except BaseException:
            # A refused replay publishes nothing, so the staged array writes
            # must not outlive it either.
            self.session.discard_changes()
            raise
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: PASS, no lint findings.

- [ ] **Step 6: Commit**

```bash
git add src/icechest/transaction.py tests/test_commit_bookkeeping.py
git commit -m "fix: commit table-only transactions and discard on failure"
```

---

### Task 8: Read-only catalogs for snapshot reads

`HybridSnapshot.table()` hands back a table whose catalog will happily write a new `metadata.json` that no commit will ever reference — an orphan in a warehouse configured never to collect anything.

**Files:**
- Modify: `src/icechest/catalog.py:60-78` (`__init__`), `:109-115` (`create_table`), `:156-166` (`commit_table`)
- Modify: `src/icechest/transaction.py:227-232` (`_catalog_for`), `:291-305` (`HybridSnapshot.table`)
- Create: `tests/test_read_only.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `IcechunkCatalog(..., read_only: bool = False)` with attribute `.read_only`; `HybridRepo._catalog_for(pointers, *, read_only: bool = False)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_read_only.py`:

```python
"""Tables loaded from a snapshot are for reading."""

from __future__ import annotations

import pytest

from tests.helpers import seed


def test_snapshot_table_refuses_writes(repo):
    """Writing here would orphan a metadata.json nothing ever references."""
    seed(repo, "g1", "g2")
    table = repo.read("main").table("granules")

    with pytest.raises(NotImplementedError, match="read-only"):
        table.delete("granule_id == 'g1'")

    rows = repo.read("main").table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g1", "g2"}


def test_snapshot_table_still_reads(repo):
    seed(repo, "g1")
    assert repo.read("main").table("granules").scan().to_arrow().num_rows == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_read_only.py -q`
Expected: FAIL — `test_snapshot_table_refuses_writes` fails with `DID NOT RAISE <class 'NotImplementedError'>`

- [ ] **Step 3: Write the implementation**

In `src/icechest/catalog.py`, add the parameter to `__init__`'s signature after `properties`:

```python
        read_only: bool = False,
```

document it in the class docstring's parameter list:

```
    read_only:
        Refuse operations that would write a new ``metadata.json``. Set for
        catalogs built from a snapshot, whose writes no commit would publish.
```

and set it alongside the other attributes:

```python
        self.read_only = read_only
```

Add this method just above `create_table`:

```python
    def _assert_writable(self) -> None:
        if self.read_only:
            raise NotImplementedError(
                "This catalog was opened from a snapshot and is read-only. "
                "Table writes must go through repo.transaction(...), so the "
                "new metadata.json is published by an Icechunk commit instead "
                "of being orphaned in the warehouse."
            )
```

Call it as the first statement of `create_table` and of `commit_table`:

```python
        self._assert_writable()
```

In `src/icechest/transaction.py`, change `_catalog_for` to:

```python
    def _catalog_for(
        self, pointers: dict[str, str], *, read_only: bool = False
    ) -> IcechunkCatalog:
        return IcechunkCatalog(
            warehouse=self.warehouse,
            pointers=pointers,
            properties=self.io_properties,
            read_only=read_only,
        )
```

and in `HybridSnapshot.table`, change the catalog construction to:

```python
        catalog = self._repo._catalog_for({name: location}, read_only=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: PASS, no lint findings.

- [ ] **Step 5: Commit**

```bash
git add src/icechest/catalog.py src/icechest/transaction.py tests/test_read_only.py
git commit -m "fix: make snapshot-loaded tables read-only"
```

---

### Task 9: Conflict scenarios

Tests only. These exercise paths the earlier tasks built but did not cover: multi-writer windows, the array/table atomicity that motivates the whole design, and a concurrently created table.

**Files:**
- Modify: `tests/test_delete.py`

**Interfaces:**
- Consumes: everything from Tasks 4–7.
- Produces: nothing.

- [ ] **Step 1: Write the tests**

Append to `tests/test_delete.py`:

```python
def test_window_spans_every_intervening_commit(repo):
    """Two writers land before Anna retries; both are inside her window."""
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna retracts g1")
    anna.delete("granules", "granule_id == 'g1'")

    with repo.transaction("main", "ben ingests g8") as ben:
        ben.append("granules", granules("g8"))
    with repo.transaction("main", "carol re-ingests g1") as carol:
        carol.append("granules", granules("g1"))

    # Carol's add is two commits back from the tip: a window anchored on the
    # most recent commit alone would miss it.
    with pytest.raises(TableConflictError):
        anna.commit()


def test_disjoint_pile_up_still_replays(repo):
    seed(repo, "g1", "g2")

    anna = repo.transaction("main", "anna retracts g1")
    anna.delete("granules", "granule_id == 'g1'")

    with repo.transaction("main", "ben ingests g8") as ben:
        ben.append("granules", granules("g8"))
    with repo.transaction("main", "carol ingests g9") as carol:
        carol.append("granules", granules("g9"))

    anna.commit()

    rows = repo.read("main").table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g2", "g8", "g9"}


def test_row_and_array_region_are_atomic(repo):
    """The causally-linked case: the row and the region it describes."""
    seed(repo, "g1", "g2")
    with repo.transaction("main", "fill g1's region") as tx:
        tx.group["data"][0:10] = np.ones(10, "f4")

    anna = repo.transaction("main", "anna retracts g1 and zeroes its region")
    anna.delete("granules", "granule_id == 'g1'")
    anna.group["data"][0:10] = np.zeros(10, "f4")

    with repo.transaction("main", "ben re-ingests g1") as ben:
        ben.append("granules", granules("g1"))

    with pytest.raises(TableConflictError):
        anna.commit()

    snap = repo.read("main")
    assert snap.group["data"][0] == 1.0  # the region was not zeroed
    assert snap.table("granules").scan().to_arrow().num_rows == 3


def test_delete_against_concurrently_created_table_is_refused(repo):
    """Two writers create the same table; ours must not delete from theirs."""
    with repo.transaction("main", "seed arrays") as tx:
        tx.group.create_array("data", shape=(100,), dtype="f4", chunks=(10,))

    anna = repo.transaction("main", "anna creates and prunes")
    anna.create_table("granules", GRANULE_SCHEMA)
    anna.delete("granules", "granule_id == 'g1'")

    with repo.transaction("main", "ben creates and fills") as ben:
        ben.create_table("granules", GRANULE_SCHEMA)
        ben.append("granules", granules("g1"))

    with pytest.raises(TableConflictError):
        anna.commit()

    rows = repo.read("main").table("granules").scan().to_arrow()
    assert set(rows["granule_id"].to_pylist()) == {"g1"}
```

and extend that file's helper import to:

```python
from tests.helpers import GRANULE_SCHEMA, granules, seed
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_delete.py -q`
Expected: PASS. If `test_window_spans_every_intervening_commit` fails, the base snapshot is being re-anchored on each recovery rather than captured once — check `_capture_base_snapshot`'s early return.

- [ ] **Step 3: Run the whole suite**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: PASS, no lint findings.

- [ ] **Step 4: Commit**

```bash
git add tests/test_delete.py
git commit -m "test: cover multi-writer windows and array/table atomicity"
```

---

### Task 10: Documentation

The README describes this work as unimplemented in two places and the example only shows appends.

**Files:**
- Modify: `README.md` (the "Concurrent writing", "Garbage collection", and "Status and open questions" sections)
- Modify: `examples/concurrent_writers.py`

**Interfaces:**
- Consumes: the public API from Tasks 4–5.
- Produces: nothing.

- [ ] **Step 1: Rewrite the concurrency claim**

In `README.md`, replace the paragraph beginning "This automatic recovery is for **appends**." with:

```markdown
Appends replay unconditionally: adding rows commutes with adding other rows.
Deletes and overwrites cannot, so they are replayed only when it is provably
safe. Before adopting the winner's version, the delete's predicate is checked
against every commit that landed since the operation was planned. If nobody
added rows matching it, the operation is rebuilt on their version; if somebody
did, the transaction raises `TableConflictError` and publishes nothing — arrays
included — so the caller can re-plan against what actually changed.

The check is snapshot isolation, and it is conservative: candidate files are
judged from column statistics, so a delete can be refused when a matching row
merely might exist.
```

- [ ] **Step 2: Note the enlarged garbage-collection surface**

In `README.md`, at the end of the "Garbage collection" section, add:

```markdown
Deletes enlarge the problem. They are copy-on-write, so a delete that loses a
race leaves behind the data files it rewrote as well as its abandoned
`metadata.json` and manifests. Nothing points at those files — a sweeper finds
them by listing the warehouse and subtracting what live refs reach.
```

- [ ] **Step 3: Update the status list**

In `README.md`, replace the "Non-append conflict re-planning" bullet under "Status and open questions" with:

```markdown
- **Upsert and merge-on-read deletes.** Deletes and overwrites replay under
  snapshot isolation; `upsert` is not exposed, and deletes are copy-on-write
  because PyIceberg does not write delete files.
```

- [ ] **Step 4: Show a delete in the example**

In `examples/concurrent_writers.py`, add this section before the tag step (currently step 6), and renumber the steps that follow:

```python
    print("6. A delete replays too, when nobody touched the rows it targets.")
    retract = repo.transaction("main", "retract G-100")
    retract.delete("granules", "granule_id == 'G-100'")
    with repo.transaction("main", "concurrent unrelated ingest") as other:
        other.group["reflectance"][100:110] = np.full(10, 5.0, "f4")
        other.append("granules", row("G-400", "2026-03-04", "100:110"))
    retract.commit()  # conflicts, validates the predicate, replays
    remaining = repo.read("main").table("granules").scan().to_arrow()
    print(f"   rows now:    {sorted(remaining['granule_id'].to_pylist())}\n")
```

- [ ] **Step 5: Verify the example runs**

Run: `uv run python examples/concurrent_writers.py`
Expected: runs to completion; step 6 prints `rows now: ['G-200', 'G-400']`. G-300 is added by the step that follows, so it is not present yet.

- [ ] **Step 6: Run the whole suite one more time**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: PASS, no lint findings.

- [ ] **Step 7: Commit**

```bash
git add README.md examples/concurrent_writers.py
git commit -m "docs: describe validated delete and overwrite replay"
```
