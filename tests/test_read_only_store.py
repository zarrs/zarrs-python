"""A store opened read-only must refuse writes, as zarr-python's own pipeline does.

zarr-python enforces this in the store itself -- `Store._check_writable`, reached from the
concrete store's `_set`. This pipeline never gets there: it is handed a `StoreConfig` and
builds its own Rust store, writable whatever mode the array was opened in. Without the guard
a write to a `mode="r"` array SUCCEEDS here and raises through the default pipeline.

Opened STRICT throughout, and that is what makes the assertion mean anything: zarr's own
refusal message is byte-identical to the Rust guard's, so with a fallback available these
tests would pass whether the guard fired or zarr-python served the write.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def array(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(64, dtype=np.float32)
    path = tmp_path / "a"
    zarr.create_array(path, dtype=values.dtype, shape=values.shape, chunks=(16,))[:] = (
        values
    )
    return path, values


def open_strict(path: Path, mode: str) -> zarr.Array:
    """Open with no fallback. `strict` has to be set BEFORE the open: that is when the
    pipeline decides whether it has one."""
    with zarr.config.set({"codec_pipeline.strict": True}):
        return zarr.open_array(path, mode=mode)


def test_write_to_a_read_only_array_raises(array: tuple[Path, np.ndarray]) -> None:
    path, values = array
    z = open_strict(path, "r")
    with pytest.raises(ValueError, match="read-only"):
        z[0:16] = -1.0
    np.testing.assert_array_equal(zarr.open_array(path, mode="r")[:], values)


def test_a_writable_array_still_writes(array: tuple[Path, np.ndarray]) -> None:
    path, values = array
    z = open_strict(path, "r+")
    z[0:16] = -1.0
    expected = values.copy()
    expected[0:16] = -1.0
    np.testing.assert_array_equal(zarr.open_array(path, mode="r")[:], expected)
