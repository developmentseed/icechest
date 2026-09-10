"""Reading the IFD chain: how many resolution levels a COG actually carries."""

from __future__ import annotations

import struct

import pytest

from icechest.demo.tiff import parse_ifds

IMAGE_WIDTH, IMAGE_LENGTH = 256, 257
SHORT, LONG = 3, 4


def build_tiff(sizes: list[tuple[int, int]]) -> bytes:
    """A little-endian classic TIFF with one minimal IFD per size given."""
    header = struct.pack("<2sHI", b"II", 42, 8)
    entry_count = 2
    ifd_size = 2 + entry_count * 12 + 4
    body = b""
    for index, (width, height) in enumerate(sizes):
        offset = 8 + index * ifd_size
        nxt = 0 if index == len(sizes) - 1 else offset + ifd_size
        body += struct.pack("<H", entry_count)
        body += struct.pack("<HHII", IMAGE_WIDTH, LONG, 1, width)
        body += struct.pack("<HHII", IMAGE_LENGTH, LONG, 1, height)
        body += struct.pack("<I", nxt)
    return header + body


def test_reads_every_ifd_in_order():
    data = build_tiff([(3660, 3660), (1830, 1830), (915, 915)])
    assert parse_ifds(data) == [(3660, 3660), (1830, 1830), (915, 915)]


def test_single_ifd_file():
    assert parse_ifds(build_tiff([(100, 200)])) == [(100, 200)]


def test_short_typed_dimensions_are_read():
    """Small images store dimensions as SHORT rather than LONG."""
    header = struct.pack("<2sHI", b"II", 42, 8)
    body = struct.pack("<H", 2)
    body += struct.pack("<HHIHH", IMAGE_WIDTH, SHORT, 1, 640, 0)
    body += struct.pack("<HHIHH", IMAGE_LENGTH, SHORT, 1, 480, 0)
    body += struct.pack("<I", 0)
    assert parse_ifds(header + body) == [(640, 480)]


def test_rejects_non_tiff():
    with pytest.raises(ValueError, match="not a TIFF"):
        parse_ifds(b"PK\x03\x04not a tiff at all")


def test_stops_when_the_chain_leaves_the_fetched_header():
    """A truncated header yields the levels it covers, not an exception.

    Only the first IFD fits in these bytes, and the second one's offset points
    past the end -- exactly what a too-small range request would produce.
    """
    data = build_tiff([(3660, 3660), (1830, 1830)])
    first_ifd_end = 8 + (2 + 2 * 12 + 4)
    assert parse_ifds(data[:first_ifd_end]) == [(3660, 3660)]
