"""Convention attributes, derived from the STAC record alone."""

from __future__ import annotations

from icechest.demo.conventions import granule_attrs, multiscale_layout

# A real record: HLS.L30.T20JKP.2026004T142004.v2.0
TRANSFORM = [30.0, 0.0, 199980.0, 0.0, -30.0, -3099960.0, 0.0, 0.0, 1.0]
SHAPE = [3660, 3660]


def test_projected_bbox_is_derived_from_transform_and_shape():
    attrs = granule_attrs(epsg=32620, shape=SHAPE, transform=TRANSFORM, levels=5)
    # xmin/ymax come straight off the transform; the other corner is one
    # grid-width away. No reprojection of the geographic bbox is involved.
    assert attrs["spatial:bbox"] == [199980.0, -3209760.0, 309780.0, -3099960.0]


def test_transform_keeps_only_the_affine_terms():
    attrs = granule_attrs(epsg=32620, shape=SHAPE, transform=TRANSFORM, levels=5)
    assert attrs["spatial:transform"] == TRANSFORM[:6]


def test_shape_is_the_pixel_counts_and_dimensions_are_their_names():
    """The schema types ``spatial:dimensions`` as the two dimension *names*.
    Pixel counts there left a reader after the grid size finding nothing and a
    reader after the names finding integers."""
    attrs = granule_attrs(epsg=32620, shape=SHAPE, transform=TRANSFORM, levels=5)
    assert attrs["spatial:shape"] == [3660, 3660]
    assert attrs["spatial:dimensions"] == ["y", "x"]
    assert attrs["proj:code"] == "EPSG:32620"


def test_three_conventions_are_declared():
    attrs = granule_attrs(epsg=32620, shape=SHAPE, transform=TRANSFORM, levels=5)
    names = [c["name"] for c in attrs["zarr_conventions"]]
    assert names == ["multiscales", "proj", "spatial"]
    uuids = {c["uuid"] for c in attrs["zarr_conventions"]}
    assert "d35379db-88df-4056-af3a-620245f8e347" in uuids


def test_layout_transform_is_the_step_from_the_level_it_derives_from():
    """A layout ``transform`` is relative to ``derived_from``, so it is the
    factor-of-two step between adjacent levels -- not the level's absolute
    affine, which is not a valid layout transform at all."""
    layout = multiscale_layout(levels=5)
    assert len(layout) == 5
    assert layout[0] == {"asset": "multiscales/0"}
    assert layout[1] == {
        "asset": "multiscales/1",
        "derived_from": "multiscales/0",
        "transform": {"scale": [2, 2]},
    }
    assert layout[4]["derived_from"] == "multiscales/3"
    assert layout[4]["transform"] == {"scale": [2, 2]}
    assert all("factors" not in entry for entry in layout)


def test_single_level_layout_has_no_derived_entries():
    assert multiscale_layout(levels=1) == [{"asset": "multiscales/0"}]


def test_layout_paths_are_relative_to_the_group_carrying_the_attributes():
    """The attributes go on the band group; the levels live one below it, in
    the ``multiscales`` group. A bare ``"0"`` would name a sibling that does
    not exist."""
    layout = multiscale_layout(levels=3)
    assert [entry["asset"] for entry in layout] == [
        "multiscales/0",
        "multiscales/1",
        "multiscales/2",
    ]
