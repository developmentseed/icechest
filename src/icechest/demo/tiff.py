"""Just enough TIFF to count a COG's resolution levels.

VirtualiZarr parses one IFD at a time, so something has to say how many there
are. Reading the chain ourselves is deterministic and needs one range request,
where probing IFD indices until the parser errors would cost a failed remote
read per granule and depend on which exception that parser happens to raise.
"""

from __future__ import annotations

import struct

#: COGs put their IFD chain at the front of the file, so a single range request
#: of this size reaches every level's header.
HEADER_BYTES = 512 * 1024

_IMAGE_WIDTH = 256
_IMAGE_LENGTH = 257
_TYPE_SHORT = 3
_TYPE_LONG = 4


def parse_ifds(header: bytes) -> list[tuple[int, int]]:
    """Return ``(width, height)`` for each IFD reachable in ``header``.

    Walks the chain until it ends or leaves the bytes we were given; a truncated
    header yields the levels it covers rather than reading past the end.
    """
    if len(header) < 8 or header[:2] not in (b"II", b"MM"):
        raise ValueError("not a TIFF: missing byte-order marker")
    endian = "<" if header[:2] == b"II" else ">"
    version, offset = struct.unpack_from(endian + "HI", header, 2)
    if version != 42:
        raise ValueError(f"not a TIFF: unsupported version {version}")

    sizes: list[tuple[int, int]] = []
    while offset and offset + 2 <= len(header):
        (count,) = struct.unpack_from(endian + "H", header, offset)
        end = offset + 2 + count * 12
        if end + 4 > len(header):
            break
        dimensions: dict[int, int] = {}
        for index in range(count):
            entry = offset + 2 + index * 12
            tag, kind = struct.unpack_from(endian + "HH", header, entry)
            if tag in (_IMAGE_WIDTH, _IMAGE_LENGTH):
                fmt = "H" if kind == _TYPE_SHORT else "I"
                (dimensions[tag],) = struct.unpack_from(endian + fmt, header, entry + 8)
        if _IMAGE_WIDTH in dimensions and _IMAGE_LENGTH in dimensions:
            sizes.append((dimensions[_IMAGE_WIDTH], dimensions[_IMAGE_LENGTH]))
        (offset,) = struct.unpack_from(endian + "I", header, end)
    return sizes
