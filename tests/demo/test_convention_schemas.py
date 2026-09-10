"""What ``granule_attrs`` produces, judged by the conventions' own schemas.

These attributes are the only georeferencing the written groups carry, so a
reader that cannot validate them cannot use them. ``schemas/`` holds byte-for-byte
copies of the three published schemas at their ``v0.1`` tag::

    https://raw.githubusercontent.com/zarr-conventions/{multiscales,spatial,proj}/refs/tags/v0.1/schema.json

vendored so the default suite needs no network, the same way the repo's own
convention is validated in ``tests/test_schema.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validator_for

from icechest.demo.conventions import (
    MULTISCALES_CONVENTION,
    PROJ_CONVENTION,
    SPATIAL_CONVENTION,
    granule_attrs,
)

SCHEMAS = Path(__file__).parent / "schemas"
TRANSFORM = [30.0, 0.0, 199980.0, 0.0, -30.0, -3099960.0, 0.0, 0.0, 1.0]
SHAPE = [3660, 3660]

#: Each convention, with the metadata object this module declares for it.
CONVENTIONS = {
    "multiscales": MULTISCALES_CONVENTION,
    "spatial": SPATIAL_CONVENTION,
    "proj": PROJ_CONVENTION,
}


def schema(name: str) -> dict:
    return json.loads((SCHEMAS / f"{name}.json").read_text())


def validator(name: str, pointer: str | None = None):
    """A validator for one schema, or for one ``$defs`` entry inside it."""
    document = schema(name)
    subschema = document if pointer is None else document["$defs"][pointer]
    return validator_for(document)(subschema)


def node(levels: int = 5) -> dict:
    """The Zarr node the writer produces: a band group carrying the attributes."""
    return {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": granule_attrs(
            epsg=32620, shape=SHAPE, transform=TRANSFORM, levels=levels
        ),
    }


@pytest.mark.parametrize("name", list(CONVENTIONS))
def test_schema_is_itself_valid(name):
    document = schema(name)
    validator_for(document).check_schema(document)


@pytest.mark.parametrize("name", list(CONVENTIONS))
def test_written_attributes_validate(name):
    validator(name).validate(node())


@pytest.mark.parametrize("name,metadata", CONVENTIONS.items())
def test_convention_metadata_matches_the_schema_exactly(name, metadata):
    """Each schema fixes every field of its metadata object as a ``const`` and
    closes the object, so a stale URL, a trailing colon on the name, or one
    extra key is a validation error rather than a harmless difference."""
    validator(name, "conventionMetadata").validate(metadata)


@pytest.mark.parametrize(
    "name,change",
    [
        ("proj", {"name": "proj:"}),
        ("spatial", {"name": "spatial:"}),
        (
            "multiscales",
            {
                "schema_url": "https://raw.githubusercontent.com/zarr-conventions/"
                "multiscales/refs/tags/v1/schema.json"
            },
        ),
        ("spatial", {"version": "0.1"}),
    ],
)
def test_metadata_object_is_closed_and_fixed(name, change):
    """Guards the exactness the test above depends on: were the objects open or
    the fields free-form, that test would pass on anything."""
    with pytest.raises(ValidationError):
        validator(name, "conventionMetadata").validate({**CONVENTIONS[name], **change})


def test_pixel_counts_in_dimensions_are_rejected():
    """The correction this test file exists to pin: ``spatial:dimensions`` is
    typed as the two dimension *names*."""
    document = node()
    document["attributes"]["spatial:dimensions"] = [3660, 3660]
    with pytest.raises(ValidationError):
        validator("spatial").validate(document)


def test_a_flat_layout_transform_is_rejected():
    """The other correction: a layout ``transform`` is an object of relative
    scale and translation, not an absolute six-element affine."""
    document = node()
    document["attributes"]["multiscales"]["layout"][1]["transform"] = [
        60.0,
        0.0,
        199980.0,
        0.0,
        -60.0,
        -3099960.0,
    ]
    with pytest.raises(ValidationError):
        validator("multiscales").validate(document)


def test_a_derived_level_without_a_transform_is_rejected():
    document = node()
    del document["attributes"]["multiscales"]["layout"][1]["transform"]
    with pytest.raises(ValidationError):
        validator("multiscales").validate(document)


def test_a_single_level_granule_still_validates():
    for name in CONVENTIONS:
        validator(name).validate(node(levels=1))
