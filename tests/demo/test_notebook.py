"""The demo notebook has to agree with itself about the access mode.

Nothing executes the notebook offline, so a cell that fetches through one
endpoint while asking for another's URL fails only in front of whoever is
running the demo. That is exactly what happened: the registry cell was updated
to take the access mode and the ``asset_urls`` call beside it was not, so it
resolved an ``s3://`` URL against an https-only registry.
"""

from __future__ import annotations

import json
import pathlib
import re

NOTEBOOK = pathlib.Path(__file__).resolve().parents[2] / "examples" / "hls_ingest.ipynb"

#: Every call whose behaviour depends on which endpoint is in use.
ACCESS_DEPENDENT = ("asset_urls", "object_store_registry", "open_store", "ingest_batch")


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text())
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_access_is_defined_before_it_is_used():
    """A later cell defining it is no help to an earlier cell using it."""
    cells = code_cells()
    defined = next(
        (i for i, src in enumerate(cells) if re.search(r"^ACCESS\s*=", src, re.M)), None
    )
    assert defined is not None, "no cell sets ACCESS"

    used = [i for i, src in enumerate(cells) if re.search(r"\bACCESS\b", src)]
    assert min(used) >= defined, (
        f"ACCESS is used in cell {min(used)} but not set until cell {defined}"
    )


def arguments(source: str, open_paren: int) -> str:
    """The text between a call's parentheses, nesting included.

    Splitting on the first ``)`` would stop at an inner call's paren and miss
    arguments after it.
    """
    depth, start = 1, open_paren + 1
    for index in range(start, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                return source[start:index]
    return source[start:]


def test_every_access_dependent_call_is_given_the_mode():
    """A default is the wrong thing here: it silently disagrees with whatever
    the rest of the notebook was set to."""
    problems = []
    for index, source in enumerate(code_cells()):
        for call in ACCESS_DEPENDENT:
            for match in re.finditer(rf"\b{call}\(", source):
                if "ACCESS" not in arguments(source, match.end() - 1):
                    problems.append(f"cell {index}: {call}() without access=ACCESS")
    assert not problems, "\n".join(problems)
