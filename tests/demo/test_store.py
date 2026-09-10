"""Store setup: the container is declared before anything writes to it."""

from __future__ import annotations

import pickle

from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType, TimestamptzType

from icechest.demo.assets import LPDAAC_S3_PREFIX
from icechest.demo.credentials import lpdaac_credentials, virtual_chunk_container
from icechest.demo.store import ensure_table, granules_schema, open_store

SOURCE = Schema(
    NestedField(field_id=1, name="id", field_type=StringType(), required=False),
    NestedField(
        field_id=2, name="datetime", field_type=TimestamptzType(), required=False
    ),
)


def test_credential_callable_is_picklable():
    """icechunk requires it: refreshable credentials may cross a process."""
    assert pickle.loads(pickle.dumps(lpdaac_credentials)) is lpdaac_credentials


def test_container_covers_the_lpdaac_prefix():
    assert virtual_chunk_container().url_prefix == LPDAAC_S3_PREFIX


def test_granules_schema_is_the_source_plus_array_path():
    schema = granules_schema(SOURCE)
    names = [field.name for field in schema.fields]
    assert names == ["id", "datetime", "array_path"]


def test_open_store_needs_no_network(tmp_path):
    """Refreshable credentials are lazy, so a store opens offline."""
    repo = open_store(tmp_path)
    assert repo.read("main").pointers == {}


def test_ensure_table_creates_once(tmp_path):
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)
    first = repo.read("main").snapshot_id
    assert "granules" in repo.read("main").pointers

    ensure_table(repo, SOURCE)
    assert repo.read("main").snapshot_id == first  # no second commit


def test_ensure_table_declares_the_array_path_column(tmp_path):
    repo = open_store(tmp_path)
    ensure_table(repo, SOURCE)
    table = repo.read("main").table("granules")
    assert "array_path" in [field.name for field in table.schema().fields]


def test_source_schema_may_not_collide_with_the_array_path_id():
    """The 'well clear of the archive's ids' comment, made self-checking: two
    fields sharing an id is a schema whose columns cannot be told apart."""
    import pytest

    from icechest.demo.store import ARRAY_PATH_FIELD_ID

    colliding = Schema(
        NestedField(
            field_id=ARRAY_PATH_FIELD_ID,
            name="id",
            field_type=StringType(),
            required=False,
        )
    )
    with pytest.raises(ValueError, match="array_path"):
        granules_schema(colliding)
