"""Opening the local Icechunk store this demo writes into."""

from __future__ import annotations

from pathlib import Path

import icechunk
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType

from icechest import HybridRepo
from icechest.demo.assets import LPDAAC_S3_PREFIX
from icechest.demo.credentials import lpdaac_credentials, virtual_chunk_container

#: Well clear of the archive's own field ids, which run into the low hundreds.
ARRAY_PATH_FIELD_ID = 1000


def granules_schema(source: Schema) -> Schema:
    """The archive's schema, plus the link to a granule's arrays.

    Kept verbatim otherwise: curating a subset would reintroduce exactly the
    bespoke metadata modelling this project exists to avoid.
    """
    return Schema(
        *source.fields,
        NestedField(
            field_id=ARRAY_PATH_FIELD_ID,
            name="array_path",
            field_type=StringType(),
            required=False,
        ),
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
        tx.create_table(name, granules_schema(source_schema))
