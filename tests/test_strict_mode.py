"""Tests for strict mode configuration.

This module tests the codec_pipeline.strict configuration option that controls
whether ZarrsCodecPipeline raises exceptions or falls back to BatchedCodecPipeline
for unsupported operations.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr
from zarr.storage import StorePath

from zarrs.pipeline import UnsupportedIndexTypeError

if TYPE_CHECKING:
    from zarr.abc.store import Store


@pytest.mark.parametrize("codec_pipeline_strict", [False, True])
class TestStrictMode:
    """Tests that verify behavior with strict mode enabled (exceptions) and disabled (fallback)."""

    @pytest.fixture(autouse=True)
    def setup(self, *, codec_pipeline_strict: bool):
        """Configure strict mode for these tests."""
        zarr.config.set({"codec_pipeline.strict": codec_pipeline_strict})
        yield
        # Reset to default after test
        zarr.config.set({"codec_pipeline.strict": False})

    @pytest.mark.filterwarnings(
        "ignore:Array is unsupported by ZarrsCodecPipeline:UserWarning"
    )
    @pytest.mark.parametrize("store", ["local"], indirect=["store"])
    def test_unsupported_dtype(
        self, store: Store, *, codec_pipeline_strict: bool
    ) -> None:
        """Test that unsupported dtypes raise or fall back based on strict mode.

        Variable-length data types are supported by zarrs, but not zarrs-python.
        zarrs-python errors out on read/write rather than array creation.
        """
        data = np.array(["hello", "world"], dtype=object)

        sp = StorePath(store, path="vlen_test")
        arr = zarr.create_array(
            sp,
            shape=data.shape,
            chunks=data.shape,
            dtype=str,
            fill_value="",
        )

        with (
            pytest.raises(UnsupportedIndexTypeError)
            if codec_pipeline_strict
            else nullcontext()
        ):
            arr[:] = data
        if not codec_pipeline_strict:
            result = arr[:]
            assert np.array_equal(result, data)

    @pytest.mark.parametrize("store", ["local"], indirect=["store"])
    def test_advanced_indexing(
        self, store: Store, *, codec_pipeline_strict: bool
    ) -> None:
        """Test that advanced indexing raises or falls back based on strict mode.

        Strided and fancy indexing are unsupported by zarrs-python.
        """
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

        with (
            pytest.raises(UnsupportedIndexTypeError)
            if codec_pipeline_strict
            else nullcontext()
        ):
            arr[:, ::2] = data[:, ::2]
            arr[[0, 2], [0, 1]] = data[[0, 2], [0, 1]]

    @pytest.mark.parametrize("store", ["local"], indirect=["store"])
    def test_supported_operations_still_work(
        self, store: Store, *, codec_pipeline_strict: bool
    ) -> None:
        """Test that supported operations work in both strict and non-strict modes."""
        sp = StorePath(store, path="normal_test")
        arr = zarr.create_array(
            sp,
            shape=(10, 10),
            chunks=(5, 5),
            dtype=np.float64,
            fill_value=0.0,
        )

        data = np.arange(100).reshape(10, 10).astype(np.float64)

        arr[:] = data
        assert np.array_equal(arr[:], data)

        # Contiguous slicing should work
        assert np.array_equal(arr[0:5, 0:5], data[0:5, 0:5])

        # Integer indexing should work
        assert arr[5, 5] == data[5, 5]

        # Contiguous array indexing should work
        indices = np.array([1, 2, 3])
        assert np.array_equal(arr[indices, 0], data[indices, 0])
