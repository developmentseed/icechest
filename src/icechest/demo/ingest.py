"""Publishing a batch of granules: metadata and arrays, in one commit."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from icechest.demo.assets import DEFAULT_BANDS
from icechest.demo.virtualize import write_granule

logger = logging.getLogger(__name__)


class BatchFailed(Exception):
    """Every granule in the batch failed, so there was nothing to publish."""


@dataclass
class BatchResult:
    snapshot_id: str
    committed: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)


def ingest_batch(
    repo: Any,
    rows: pa.Table,
    *,
    registry: Any,
    bands: Sequence[str] = DEFAULT_BANDS,
    writer: Callable[..., str] = write_granule,
    message: str | None = None,
    name: str = "granules",
) -> BatchResult:
    """Stage every granule's arrays, append their rows, and commit once.

    A granule that fails contributes neither arrays nor a row, so the invariant
    this store exists for still holds exactly: a row and its arrays are always
    published together. Aborting the whole batch for one unreadable COG would
    repeat all the work on retry, which is the wrong trade at archive scale.
    """
    tx = repo.transaction("main", message or f"ingest {rows.num_rows} granules")
    committed: list[str] = []
    skipped: dict[str, str] = {}
    keep: list[dict[str, Any]] = []

    for row in rows.to_pylist():
        try:
            array_path = writer(tx, row, registry=registry, bands=bands)
        except Exception as error:  # noqa: BLE001 - any failure skips one granule
            logger.info("skipping %s: %s", row["id"], error)
            skipped[row["id"]] = str(error)
            continue
        keep.append({**row, "array_path": array_path})
        committed.append(row["id"])

    if not keep:
        tx.session.discard_changes()
        raise BatchFailed(f"all {rows.num_rows} granules failed: {skipped}")

    schema = repo.read("main").table(name).schema().as_arrow()
    tx.append(name, pa.Table.from_pylist(keep, schema=schema))
    snapshot_id = tx.commit()
    return BatchResult(snapshot_id=snapshot_id, committed=committed, skipped=skipped)
