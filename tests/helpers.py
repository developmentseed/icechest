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
