"""Opening the local Icechunk store this demo writes into."""

from __future__ import annotations

from pathlib import Path

import icechunk
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table.sorting import SortField, SortOrder
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType

from icechest import HybridRepo
from icechest.demo.assets import LPDAAC_S3_PREFIX
from icechest.demo.credentials import lpdaac_credentials, virtual_chunk_container
from icechest.demo.hash import HASH_BLOCK_COLUMN, HASH_COLUMN

#: Well clear of the archive's own field ids, which run into the low hundreds.
#: ``granules_schema`` checks that claim against the schema it is handed, since
#: two fields sharing an id is a schema whose columns cannot be told apart.
ARRAY_PATH_FIELD_ID = 1000
STAC_HASH_FIELD_ID = 1001
STAC_HASH_BLOCK_FIELD_ID = 1002

#: Iceberg needs a partition field id of its own, in its own numbering space.
PARTITION_FIELD_ID = 1100


def granules_schema(source: Schema) -> Schema:
    """The archive's schema, plus the link to a granule's arrays and its hash.

    Kept verbatim otherwise: curating a subset would reintroduce exactly the
    bespoke metadata modelling this project exists to avoid.
    """
    if source.highest_field_id >= ARRAY_PATH_FIELD_ID:
        raise ValueError(
            f"the source schema uses field id {source.highest_field_id}, which "
            f"collides with array_path's {ARRAY_PATH_FIELD_ID}; pick a higher id"
        )
    return Schema(
        *source.fields,
        NestedField(
            field_id=ARRAY_PATH_FIELD_ID,
            name="array_path",
            field_type=StringType(),
            required=False,
        ),
        NestedField(
            field_id=STAC_HASH_FIELD_ID,
            name=HASH_COLUMN,
            field_type=LongType(),
            required=False,
        ),
        NestedField(
            field_id=STAC_HASH_BLOCK_FIELD_ID,
            name=HASH_BLOCK_COLUMN,
            field_type=LongType(),
            required=False,
        ),
    )


def granules_sort_order() -> SortOrder:
    """Ascending by hash, so a file holds a contiguous spatio-temporal range.

    Declarative only: PyIceberg's write path does not consult a sort order, so
    the rows are sorted before they are appended. This records the intent for
    compactors and readers, and makes the metadata true.
    """
    return SortOrder(
        SortField(source_id=STAC_HASH_FIELD_ID, transform=IdentityTransform())
    )


def granules_partition_spec() -> PartitionSpec:
    """One partition per block of the hash space.

    A partition is therefore a contiguous spatio-temporal region, and prunes on
    a bbox or datetime predicate. Bucket would hash the value a second time and
    scatter adjacent granules across partitions, throwing away the locality the
    Morton code exists to provide.

    The block arrives precomputed as its own column, so the transform here is
    identity. Iceberg's ``truncate`` would be the natural way to express "the
    high-order bits of the hash", but its width is a 32-bit parameter: against
    a 63-bit code the widest block it can describe still leaves billions of
    partitions.
    """
    return PartitionSpec(
        PartitionField(
            source_id=STAC_HASH_BLOCK_FIELD_ID,
            field_id=PARTITION_FIELD_ID,
            transform=IdentityTransform(),
            name=HASH_BLOCK_COLUMN,
        )
    )


def open_store(path: Path | str, *, warehouse: str | None = None) -> HybridRepo:
    """Open or create the store, with the LP DAAC container declared.

    The container has to exist before any virtual reference is written --
    ``to_icechunk`` validates it -- so it is configured at open time rather than
    at write time. Credentials are refreshed lazily, so this needs no network.
    """
    path = Path(path)
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(virtual_chunk_container())
    repo = icechunk.Repository.open_or_create(
        icechunk.local_filesystem_storage(str(path / "icechunk")),
        config=config,
        authorize_virtual_chunk_access=icechunk.containers_credentials(
            {
                LPDAAC_S3_PREFIX: icechunk.s3_refreshable_credentials(
                    get_credentials=lpdaac_credentials
                )
            }
        ),
    )
    return HybridRepo(repo, warehouse or str(path / "warehouse"))


def ensure_table(
    repo: HybridRepo, source_schema: Schema, *, name: str = "granules"
) -> None:
    """Create the granules table if this store does not have it yet.

    Checks the commit metadata rather than the Zarr attributes, so it works on a
    store whose root group has not been written yet.
    """
    if name in repo.read("main").pointers:
        return
    with repo.transaction("main", f"create {name} table") as tx:
        tx.create_table(
            name,
            granules_schema(source_schema),
            partition_spec=granules_partition_spec(),
            sort_order=granules_sort_order(),
        )
