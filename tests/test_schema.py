"""Validate the convention's examples against its published JSON schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

CONVENTION = Path(__file__).parent.parent / "conventions" / "icechunk-iceberg"
SCHEMA = json.loads((CONVENTION / "schema.json").read_text())
EXAMPLES = sorted((CONVENTION / "examples").glob("*.json"))


def test_schema_is_itself_valid():
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_validates(path):
    Draft202012Validator(SCHEMA).validate(json.loads(path.read_text()))


def test_binding_rejects_unknown_fields():
    """The binding is closed: a stray 'version_source' is a schema error."""
    doc = {
        "attributes": {
            "iceberg:version": "1",
            "iceberg:tables": {
                "t": {"location": "/w/t", "version_source": "attribute"}
            },
        }
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(SCHEMA).validate(doc)


def test_binding_requires_location():
    doc = {"attributes": {"iceberg:version": "1", "iceberg:tables": {"t": {}}}}
    with pytest.raises(ValidationError, match="location"):
        Draft202012Validator(SCHEMA).validate(doc)


def test_python_implementation_matches_published_examples():
    """The constants in code must not drift from the spec's examples."""
    from icechest.convention import CONVENTION_UUID, CONVENTION_VERSION

    minimal = json.loads((CONVENTION / "examples" / "minimal_example.json").read_text())
    attrs = minimal["attributes"]
    assert attrs["iceberg:version"] == CONVENTION_VERSION
    assert attrs["zarr_conventions"][0]["uuid"] == CONVENTION_UUID


def test_written_store_validates_against_schema(tmp_path):
    """What the implementation actually writes must satisfy the published schema."""
    import icechunk
    import pyarrow as pa

    from icechest import HybridRepo

    repo = HybridRepo.create(
        icechunk.local_filesystem_storage(str(tmp_path / "ic")),
        warehouse=str(tmp_path / "wh"),
    )
    schema = pa.schema([pa.field("granule_id", pa.string(), nullable=False)])
    with repo.transaction("main", "seed") as tx:
        tx.group.create_array("data", shape=(4,), dtype="f4", chunks=(4,))
        tx.create_table("granules", schema)

    attrs = dict(repo.read("main").group.attrs)
    Draft202012Validator(SCHEMA).validate({"attributes": attrs})
