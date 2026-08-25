"""The Anna/Ben scenario from NASA-IMPACT/veda-odd#460, end to end.

Two writers ingest different granules at the same time. Ben commits first.
Anna's commit conflicts, so her Iceberg operation is replayed onto Ben's table
version and her staged array chunks are rebased onto his snapshot. The final
history is v10 -> v11-ben -> v12-anna, with both writers' arrays and both
writers' rows present.

Run with:  uv run python examples/concurrent_writers.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import icechunk
import numpy as np
import pyarrow as pa

from icechest import HybridRepo
from icechest.pointer import POINTER_KEY

GRANULES = pa.schema(
    [
        pa.field("granule_id", pa.string(), nullable=False),
        pa.field("datetime", pa.timestamp("ms"), nullable=False),
        pa.field("array_slice", pa.string(), nullable=False),
    ]
)


def row(granule_id: str, day: str, array_slice: str) -> pa.Table:
    return pa.table(
        {
            "granule_id": [granule_id],
            "datetime": pa.array([np.datetime64(day, "ms")], pa.timestamp("ms")),
            "array_slice": [array_slice],
        },
        schema=GRANULES,
    )


def main() -> None:
    root = Path(tempfile.mkdtemp())
    repo = HybridRepo.create(
        icechunk.local_filesystem_storage(str(root / "icechunk")),
        warehouse=str(root / "warehouse"),
    )

    print("1. Seed the store: one array, one table, declared via the convention.")
    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("reflectance", shape=(400,), dtype="f4", chunks=(10,))
        tx.create_table("granules", GRANULES)
    print(f"   bindings: {repo.read('main').bindings}\n")

    print("2. Ben and Anna both open a transaction against the same tip.")
    ben = repo.transaction("main", "ben ingests G-100")
    anna = repo.transaction("main", "anna ingests G-200")

    ben.group["reflectance"][0:10] = np.ones(10, "f4")
    ben.append("granules", row("G-100", "2026-03-01", "0:10"))

    anna.group["reflectance"][200:210] = np.full(10, 2.0, "f4")
    anna.append("granules", row("G-200", "2026-03-02", "200:210"))

    print("3. Ben commits first.")
    ben_snapshot = ben.commit()
    print(f"   ben  -> {ben_snapshot}")

    print("4. Anna commits: conflicts, replays onto Ben's version, rebases.")
    anna_snapshot = anna.commit()
    print(f"   anna -> {anna_snapshot}\n")

    print("5. Both writers' work is present in one consistent snapshot.")
    tip = repo.read("main")
    rows = tip.table("granules").scan().to_arrow()
    print(f"   table rows:  {sorted(rows['granule_id'].to_pylist())}")
    print(
        f"   arrays:      [0]={tip.group['reflectance'][0]}  "
        f"[200]={tip.group['reflectance'][200]}"
    )
    print(f"   pointer:     {Path(tip.pointers['granules']).name}\n")

    print("6. A tag pins arrays and table version together.")
    repo.repo.create_tag("release-1", anna_snapshot)
    with repo.transaction("main", "later ingest") as tx:
        tx.group["reflectance"][300:310] = np.full(10, 9.0, "f4")
        tx.append("granules", row("G-300", "2026-03-03", "300:310"))

    pinned, latest = repo.read(tag="release-1"), repo.read("main")
    print(
        f"   release-1: {pinned.table('granules').scan().to_arrow().num_rows} rows, "
        f"reflectance[300]={pinned.group['reflectance'][300]}"
    )
    print(
        f"   main:      {latest.table('granules').scan().to_arrow().num_rows} rows, "
        f"reflectance[300]={latest.group['reflectance'][300]}"
    )

    print("\n7. Icechunk history (each commit carries its table pointer):")
    for snap in repo.repo.ancestry(branch="main"):
        ptr = (snap.metadata or {}).get(POINTER_KEY, {}).get("granules")
        name = Path(ptr).name if ptr else "-"
        print(f"   {snap.id[:10]}  {snap.message[:28]:28}  {name}")


if __name__ == "__main__":
    main()
