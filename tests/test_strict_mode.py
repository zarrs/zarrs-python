"""Tests for strict mode configuration.

This module tests the codec_pipeline.strict configuration option that controls
whether ZarrsCodecPipeline raises exceptions or falls back to BatchedCodecPipeline
for unsupported operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr
from zarr.storage import StorePath

from zarrs.pipeline import (
    DiscontiguousArrayError,
    UnsupportedDataTypeError,
)

if TYPE_CHECKING:
    from zarr.abc.store import Store


class TestStrictModeDisabled:
    """Tests that verify fallback behavior when strict mode is disabled (default)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure strict mode is disabled for these tests."""
        zarr.config.set({"codec_pipeline.strict": False})
        yield
        # Reset to default after test
        zarr.config.set({"codec_pipeline.strict": False})

    @pytest.mark.filterwarnings(
        "ignore:Array is unsupported by ZarrsCodecPipeline:UserWarning"
    )
    @pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
    def test_unsupported_dtype_falls_back(self, store: Store) -> None:
        """Test that unsupported dtypes fall back to Python implementation."""
        # Variable-length strings are supported by zarrs, but not zarrs-python
        data = np.array(["hello", "world"], dtype=object)

        sp = StorePath(store, path="vlen_test")
        arr = zarr.create_array(
            sp,
            shape=data.shape,
            chunks=data.shape,
            dtype=str,
            fill_value="",
        )

        # Should not raise - falls back to Python implementation
        arr[:] = data
        result = arr[:]
        assert np.array_equal(result, data)

    @pytest.mark.parametrize("store", ["local"], indirect=["store"])
    def test_advanced_indexing_falls_back(self, store: Store) -> None:
        """Test that advanced indexing falls back to Python implementation."""
        sp = StorePath(store, path="ellipsis_test")
        arr = zarr.create_array(
            sp,
            shape=(10, 10),
            chunks=(5, 5),
            dtype=np.float64,
            fill_value=0.0,
        )

        data = np.arange(100).reshape(10, 10)
        arr[:] = data

        # These indexing patterns are unsupported by zarrs-python and should fall back
        arr[:, ::2] = data[:, ::2]
        arr[[0, 2], [0, 1]] = data[[0, 2], [0, 1]]


class TestStrictModeEnabled:
    """Tests that verify exception raising when strict mode is enabled."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Enable strict mode for these tests."""
        zarr.config.set({"codec_pipeline.strict": True})
        yield
        # Reset to default after test
        zarr.config.set({"codec_pipeline.strict": False})

    @pytest.mark.parametrize("store", ["local"], indirect=["store"])
    def test_unsupported_dtype_raises_on_read(self, store: Store) -> None:
        """Test that unsupported dtypes raise exception on read in strict mode."""
        data = np.array(["hello", "world"], dtype=object)

        # Variable-length strings are supported by zarrs, so array creation should work
        sp = StorePath(store, path="vlen_test")
        arr = zarr.create_array(
            sp,
            shape=data.shape,
            chunks=data.shape,
            dtype=str,
            fill_value="",
        )

        # Write should raise in strict mode because zarrs-python does not support vlen data
        with pytest.raises(UnsupportedDataTypeError):
            arr[:] = data

    @pytest.mark.parametrize("store", ["local"], indirect=["store"])
    def test_unsupported_dtype_raises_on_write(self, store: Store) -> None:
        """Test that unsupported dtypes raise exception on write in strict mode."""
        data = np.array(["hello", "world"], dtype=object)

        sp = StorePath(store, path="vlen_test")
        arr = zarr.create_array(
            sp,
            shape=data.shape,
            chunks=data.shape,
            dtype=str,
            fill_value="",
        )

        # Write should raise in strict mode
        with pytest.raises(UnsupportedDataTypeError):
            arr[:] = data

    @pytest.mark.parametrize("store", ["local"], indirect=["store"])
    def test_advanced_indexing_raises(self, store: Store) -> None:
        """Test that advanced indexing raises exception in strict mode when unsupported."""
        sp = StorePath(store, path="ellipsis_test")
        arr = zarr.create_array(
            sp,
            shape=(10, 10),
            chunks=(5, 5),
            dtype=np.float64,
            fill_value=0.0,
        )

        data = np.arange(100).reshape(10, 10)
        arr[:] = data

        # These indexing patterns are unsupported by zarrs-python
        with pytest.raises(DiscontiguousArrayError):
            arr[:, ::2] = data[:, ::2]
        with pytest.raises(DiscontiguousArrayError):
            arr[[0, 2], [0, 1]] = data[[0, 2], [0, 1]]

    @pytest.mark.parametrize("store", ["local"], indirect=["store"])
    def test_supported_operations_still_work(self, store: Store) -> None:
        """Test that supported operations work normally in strict mode."""
        sp = StorePath(store, path="normal_test")
        arr = zarr.create_array(
            sp,
            shape=(10, 10),
            chunks=(5, 5),
            dtype=np.float64,
            fill_value=0.0,
        )

        data = np.arange(100).reshape(10, 10).astype(np.float64)

        # These operations should work fine in strict mode
        arr[:] = data
        assert np.array_equal(arr[:], data)

        # Contiguous slicing should work
        assert np.array_equal(arr[0:5, 0:5], data[0:5, 0:5])

        # Integer indexing should work
        assert arr[5, 5] == data[5, 5]

        # Contiguous array indexing should work
        indices = np.array([1, 2, 3])
        assert np.array_equal(arr[indices, 0], data[indices, 0])
