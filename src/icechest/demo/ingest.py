"""Publishing a batch of granules: metadata and arrays, in one commit."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import zarr
from zarr.core.sync import sync
from zarr.errors import NodeNotFoundError

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


def _granule_exists(store: Any, granule_id: str) -> bool:
    """Is there already a node at ``/{granule_id}`` in this session?

    Deliberately asked of the store rather than inferred from an exception the
    write raised: whether a re-write raises at all depends on ``to_icechunk``'s
    mode, so a check keyed off an error type would go quiet the moment that
    mode changed -- which is exactly what happened when the levels moved to
    ``mode="a"``. Any node counts, not just a group: the question is whether
    deleting this path could destroy something, and only "nothing is there" is
    an answer that makes it safe.
    """
    try:
        zarr.open(store=store, path=granule_id, mode="r")
    except NodeNotFoundError:
        return False
    return True


def _record_skip(skipped: dict[str, str], granule_id: str, reason: str) -> None:
    """Record why a granule was skipped, keeping any earlier reason.

    Ids are not guaranteed unique within a batch, and a reason silently
    replaced is a reason lost.
    """
    logger.info("skipping %s: %s", granule_id, reason)
    previous = skipped.get(granule_id)
    skipped[granule_id] = f"{previous}; {reason}" if previous else reason


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

    A granule already in the store is skipped before anything is
    written. This demo only appends -- ``tx.overwrite`` would be the basis of a
    re-ingest flow, and there isn't one -- so re-writing an existing granule
    could only either duplicate its row or destroy the arrays an earlier commit
    already published, and the cleanup path below would do exactly the latter.
    """
    tx = repo.transaction("main", message or f"ingest {rows.num_rows} granules")
    committed: list[str] = []
    skipped: dict[str, str] = {}
    keep: list[dict[str, Any]] = []

    for row in rows.to_pylist():
        granule_id = row["id"]
        if granule_id in committed:
            _record_skip(
                skipped,
                granule_id,
                "already staged earlier in this batch; the first occurrence's "
                "arrays are the ones published",
            )
            continue
        if _granule_exists(tx.session.store, granule_id):
            _record_skip(
                skipped,
                granule_id,
                f"/{granule_id} is already in the store; this demo appends only, "
                "so an existing granule is left exactly as it was published",
            )
            continue

        try:
            array_path = writer(tx, row, registry=registry, bands=bands)
        except Exception as error:  # noqa: BLE001 - any failure skips one granule
            reason = str(error)
            # write_granule is not atomic: it can raise after already staging
            # some of a granule's bands into the shared session. Left in place,
            # those orphaned arrays would ride along with whatever other
            # granules this batch does commit.
            #
            # The delete is safe only because of the check above: /{granule_id}
            # did not exist when this iteration began, so anything there now was
            # staged by this batch. A failure raised before any write -- an
            # unreadable COG header, a shape mismatch -- leaves nothing there and
            # cleans nothing up. A cleanup failure must not itself abort the
            # batch, so it is folded into the skip reason.
            if _granule_exists(tx.session.store, granule_id):
                try:
                    sync(tx.session.store.delete_dir(granule_id))
                except Exception as cleanup_error:  # noqa: BLE001
                    reason = f"{reason} (cleanup also failed: {cleanup_error})"
            _record_skip(skipped, granule_id, reason)
            continue
        keep.append({**row, "array_path": array_path})
        committed.append(granule_id)

    if not keep:
        tx.session.discard_changes()
        raise BatchFailed(f"all {rows.num_rows} granules failed: {skipped}")

    schema = repo.read("main").table(name).schema().as_arrow()
    tx.append(name, pa.Table.from_pylist(keep, schema=schema))
    try:
        snapshot_id = tx.commit()
    except Exception:
        # commit() publishes nothing when it raises but leaves the session live
        # for a caller who is not using the transaction as a context manager --
        # and this one is not. Drop the staged arrays rather than leave them to
        # ride along with whatever the caller does with the repo next.
        tx.session.discard_changes()
        raise
    return BatchResult(snapshot_id=snapshot_id, committed=committed, skipped=skipped)
