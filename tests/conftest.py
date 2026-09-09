"""Shared pytest fixtures."""

from __future__ import annotations

import icechunk
import pytest

from icechest import HybridRepo


@pytest.fixture
def repo(tmp_path):
    store = icechunk.local_filesystem_storage(str(tmp_path / "icechunk"))
    return HybridRepo.create(store, warehouse=str(tmp_path / "warehouse"))
