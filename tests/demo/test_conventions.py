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


def test_dimensions_and_code():
    attrs = granule_attrs(epsg=32620, shape=SHAPE, transform=TRANSFORM, levels=5)
    assert attrs["spatial:dimensions"] == [3660, 3660]
    assert attrs["proj:code"] == "EPSG:32620"


def test_three_conventions_are_declared():
    attrs = granule_attrs(epsg=32620, shape=SHAPE, transform=TRANSFORM, levels=5)
    names = [c["name"] for c in attrs["zarr_conventions"]]
    assert names == ["multiscales", "proj:", "spatial:"]
    uuids = {c["uuid"] for c in attrs["zarr_conventions"]}
    assert "d35379db-88df-4056-af3a-620245f8e347" in uuids


def test_multiscale_layout_scales_each_level():
    layout = multiscale_layout(TRANSFORM[:6], levels=5)
    assert len(layout) == 5
    assert layout[0] == {"asset": "multiscales/0"}
    assert layout[1]["derived_from"] == "multiscales/0"
    assert layout[1]["factors"] == [2, 2]
    # Level n's pixel size is 2**n times level 0's; the origin never moves.
    assert layout[1]["transform"] == [60.0, 0.0, 199980.0, 0.0, -60.0, -3099960.0]
    assert layout[4]["transform"] == [480.0, 0.0, 199980.0, 0.0, -480.0, -3099960.0]
    assert layout[4]["derived_from"] == "multiscales/3"


def test_single_level_layout_has_no_derived_entries():
    assert multiscale_layout(TRANSFORM[:6], levels=1) == [{"asset": "multiscales/0"}]


def test_layout_paths_are_relative_to_the_group_carrying_the_attributes():
    """The attributes go on the band group; the levels live one below it, in
    the ``multiscales`` group. A bare ``"0"`` would name a sibling that does
    not exist."""
    layout = multiscale_layout(TRANSFORM[:6], levels=3)
    assert [entry["asset"] for entry in layout] == [
        "multiscales/0",
        "multiscales/1",
        "multiscales/2",
    ]
