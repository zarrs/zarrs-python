from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from zarrs import ZarrsArray


@pytest.fixture
def store(tmp_path):
    return LocalStore(str(tmp_path / "test.zarr"))


def save_array(store, data, *, chunks=None):
    z = zarr.open_array(
        store, mode="w", shape=data.shape, dtype=data.dtype, chunks=chunks
    )
    z[:] = data


def open_zarrs(store, **kwargs):
    return ZarrsArray(zarr.open_array(store), **kwargs)


class TestProperties:
    def test_shape_2d(self, store):
        save_array(store, np.zeros((10, 20), dtype="float32"))
        arr = open_zarrs(store)
        assert arr.shape == (10, 20)

    def test_shape_1d(self, store):
        save_array(store, np.zeros(50, dtype="int32"))
        arr = open_zarrs(store)
        assert arr.shape == (50,)

    def test_shape_3d(self, store):
        save_array(store, np.zeros((3, 4, 5), dtype="uint8"))
        arr = open_zarrs(store)
        assert arr.shape == (3, 4, 5)

    def test_ndim(self, store):
        save_array(store, np.zeros((3, 4, 5), dtype="float64"))
        arr = open_zarrs(store)
        assert arr.ndim == 3

    def test_size(self, store):
        save_array(store, np.zeros((3, 4, 5), dtype="float64"))
        arr = open_zarrs(store)
        assert arr.size == 60

    def test_dtype(self, store):
        save_array(store, np.zeros(5, dtype="float32"))
        arr = open_zarrs(store)
        assert arr.dtype == np.dtype("float32")


class TestDtypes:
    @pytest.mark.parametrize(
        "dtype",
        [
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float32",
            "float64",
        ],
    )
    def test_dtype_roundtrip(self, tmp_path, dtype):
        data = np.arange(12, dtype=dtype).reshape(3, 4)
        store = LocalStore(str(tmp_path / "test.zarr"))
        save_array(store, data)
        arr = open_zarrs(store)
        assert arr.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(arr[:], data)

    def test_dtype_bool(self, tmp_path):
        data = np.array([[True, False, True], [False, True, False]])
        store = LocalStore(str(tmp_path / "test.zarr"))
        save_array(store, data)
        arr = open_zarrs(store)
        assert arr.dtype == np.dtype("bool")
        np.testing.assert_array_equal(arr[:], data)


class TestFullRead:
    def test_full_read_1d(self, store):
        data = np.arange(20, dtype="int64")
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[:], data)

    def test_full_read_2d(self, store):
        data = np.arange(100, dtype="float32").reshape(10, 10)
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[:], data)

    def test_full_read_3d(self, store):
        data = np.arange(60, dtype="float64").reshape(3, 4, 5)
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[:], data)


class TestSliceRead:
    def test_slice_1d(self, store):
        data = np.arange(20, dtype="int32")
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[5:10], data[5:10])

    def test_slice_2d(self, store):
        data = np.arange(100, dtype="float32").reshape(10, 10)
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[2:5, 3:7], data[2:5, 3:7])

    def test_slice_3d(self, store):
        data = np.arange(120, dtype="float64").reshape(4, 5, 6)
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[1:3, 2:4, 0:3], data[1:3, 2:4, 0:3])

    def test_partial_dims(self, store):
        data = np.arange(100, dtype="float32").reshape(10, 10)
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[2:4], data[2:4])

    def test_across_chunks(self, store):
        data = np.arange(100, dtype="int32").reshape(10, 10)
        save_array(store, data, chunks=(3, 3))
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[1:8, 2:9], data[1:8, 2:9])


class TestIntegerIndex:
    def test_integer_2d(self, store):
        data = np.arange(30, dtype="float64").reshape(5, 6)
        save_array(store, data)
        arr = open_zarrs(store)
        result = arr[2]
        np.testing.assert_array_equal(result, data[2])
        assert result.shape == (6,)

    def test_integer_3d(self, store):
        data = np.arange(60, dtype="float64").reshape(3, 4, 5)
        save_array(store, data)
        arr = open_zarrs(store)
        result = arr[1]
        np.testing.assert_array_equal(result, data[1])
        assert result.shape == (4, 5)

    def test_integer_1d_scalar(self, store):
        data = np.arange(10, dtype="int32")
        save_array(store, data)
        arr = open_zarrs(store)
        result = arr[3]
        np.testing.assert_array_equal(result, data[3])
        assert result.shape == ()

    def test_mixed_int_and_slice(self, store):
        data = np.arange(100, dtype="float32").reshape(10, 10)
        save_array(store, data)
        arr = open_zarrs(store)
        result = arr[3, 2:5]
        np.testing.assert_array_equal(result, data[3, 2:5])
        assert result.shape == (3,)

    def test_multiple_integers(self, store):
        data = np.arange(60, dtype="float64").reshape(3, 4, 5)
        save_array(store, data)
        arr = open_zarrs(store)
        result = arr[1, 2]
        np.testing.assert_array_equal(result, data[1, 2])
        assert result.shape == (5,)


class TestNegativeIndex:
    def test_negative_integer(self, store):
        data = np.arange(10, dtype="int32")
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[-1], data[-1])

    def test_negative_slice(self, store):
        data = np.arange(10, dtype="int32")
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[-3:], data[-3:])

    def test_negative_both(self, store):
        data = np.arange(20, dtype="float64").reshape(4, 5)
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[-2:, -3:], data[-2:, -3:])


class TestEmptySlice:
    def test_empty_1d(self, store):
        data = np.arange(10, dtype="int64")
        save_array(store, data)
        arr = open_zarrs(store)
        result = arr[3:3]
        assert result.shape == (0,)
        assert result.dtype == np.int64

    def test_empty_2d(self, store):
        data = np.arange(20, dtype="float32").reshape(4, 5)
        save_array(store, data)
        arr = open_zarrs(store)
        result = arr[2:2, :]
        assert result.shape == (0, 5)


class TestEdgeCases:
    def test_single_element(self, store):
        data = np.array([42], dtype="int32")
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[:], data)

    def test_large_chunk_small_slice(self, store):
        data = np.arange(1000, dtype="float32").reshape(10, 100)
        save_array(store, data, chunks=(10, 100))
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[0:1, 0:1], data[0:1, 0:1])


class TestErrors:
    def test_too_many_indices(self, store):
        save_array(store, np.zeros((3, 4), dtype="float32"))
        arr = open_zarrs(store)
        with pytest.raises(IndexError, match="too many indices"):
            arr[0, 0, 0]

    def test_out_of_bounds_integer(self, store):
        save_array(store, np.zeros(5, dtype="float32"))
        arr = open_zarrs(store)
        with pytest.raises(IndexError, match="out of bounds"):
            arr[5]


class TestWrite:
    def test_full_write_1d(self, store):
        save_array(store, np.zeros(10, dtype="int32"))
        arr = open_zarrs(store)
        data = np.arange(10, dtype="int32")
        arr[:] = data
        np.testing.assert_array_equal(arr[:], data)

    def test_full_write_2d(self, store):
        save_array(store, np.zeros((4, 5), dtype="float64"))
        arr = open_zarrs(store)
        data = np.arange(20, dtype="float64").reshape(4, 5)
        arr[:] = data
        np.testing.assert_array_equal(arr[:], data)

    def test_full_write_3d(self, store):
        save_array(store, np.zeros((2, 3, 4), dtype="float32"))
        arr = open_zarrs(store)
        data = np.arange(24, dtype="float32").reshape(2, 3, 4)
        arr[:] = data
        np.testing.assert_array_equal(arr[:], data)


class TestWriteSlice:
    def test_slice_1d(self, store):
        data = np.zeros(10, dtype="int32")
        save_array(store, data)
        arr = open_zarrs(store)
        arr[3:7] = np.array([10, 20, 30, 40], dtype="int32")
        expected = data.copy()
        expected[3:7] = [10, 20, 30, 40]
        np.testing.assert_array_equal(arr[:], expected)

    def test_slice_2d(self, store):
        data = np.zeros((5, 6), dtype="float64")
        save_array(store, data)
        arr = open_zarrs(store)
        patch = np.ones((2, 3), dtype="float64") * 99
        arr[1:3, 2:5] = patch
        expected = data.copy()
        expected[1:3, 2:5] = 99
        np.testing.assert_array_equal(arr[:], expected)

    def test_partial_dims(self, store):
        data = np.zeros((4, 5), dtype="int64")
        save_array(store, data)
        arr = open_zarrs(store)
        patch = np.ones((2, 5), dtype="int64") * 7
        arr[1:3] = patch
        expected = data.copy()
        expected[1:3] = 7
        np.testing.assert_array_equal(arr[:], expected)

    def test_across_chunks(self, store):
        data = np.zeros((10, 10), dtype="int32")
        save_array(store, data, chunks=(3, 3))
        arr = open_zarrs(store)
        patch = np.arange(49, dtype="int32").reshape(7, 7)
        arr[1:8, 2:9] = patch
        expected = data.copy()
        expected[1:8, 2:9] = patch
        np.testing.assert_array_equal(arr[:], expected)


class TestWriteIntegerIndex:
    def test_integer_2d(self, store):
        data = np.zeros((5, 6), dtype="float64")
        save_array(store, data)
        arr = open_zarrs(store)
        row = np.arange(6, dtype="float64") + 1
        arr[2] = row
        expected = data.copy()
        expected[2] = row
        np.testing.assert_array_equal(arr[:], expected)

    def test_integer_1d_scalar(self, store):
        data = np.zeros(10, dtype="int32")
        save_array(store, data)
        arr = open_zarrs(store)
        arr[3] = 42
        expected = data.copy()
        expected[3] = 42
        np.testing.assert_array_equal(arr[:], expected)

    def test_mixed_int_and_slice(self, store):
        data = np.zeros((10, 10), dtype="float32")
        save_array(store, data)
        arr = open_zarrs(store)
        arr[3, 2:5] = np.array([10, 20, 30], dtype="float32")
        expected = data.copy()
        expected[3, 2:5] = [10, 20, 30]
        np.testing.assert_array_equal(arr[:], expected)

    def test_multiple_integers(self, store):
        data = np.zeros((3, 4, 5), dtype="float64")
        save_array(store, data)
        arr = open_zarrs(store)
        arr[1, 2] = np.arange(5, dtype="float64")
        expected = data.copy()
        expected[1, 2] = np.arange(5, dtype="float64")
        np.testing.assert_array_equal(arr[:], expected)


class TestWriteErrors:
    def test_shape_mismatch(self, store):
        save_array(store, np.zeros((4, 5), dtype="float32"))
        arr = open_zarrs(store)
        with pytest.raises(ValueError, match="could not broadcast"):
            arr[0:2, 0:3] = np.zeros((3, 2), dtype="float32")

    def test_too_many_indices(self, store):
        save_array(store, np.zeros((3, 4), dtype="float32"))
        arr = open_zarrs(store)
        with pytest.raises(IndexError, match="too many indices"):
            arr[0, 0, 0] = 1.0

    def test_out_of_bounds(self, store):
        save_array(store, np.zeros(5, dtype="float32"))
        arr = open_zarrs(store)
        with pytest.raises(IndexError, match="out of bounds"):
            arr[5] = 1.0

    def test_dtype_coercion(self, store):
        data = np.zeros(5, dtype="float64")
        save_array(store, data)
        arr = open_zarrs(store)
        arr[0:3] = np.array([1, 2, 3], dtype="int32")
        expected = data.copy()
        expected[0:3] = [1, 2, 3]
        np.testing.assert_array_equal(arr[:], expected)


class TestLazyCopy:
    def test_full_copy(self, tmp_path):
        data = np.arange(20, dtype="float64").reshape(4, 5)
        src_store = LocalStore(str(tmp_path / "src.zarr"))
        dst_store = LocalStore(str(tmp_path / "dst.zarr"))
        save_array(src_store, data)
        save_array(dst_store, np.zeros_like(data))
        src = open_zarrs(src_store)
        dst = open_zarrs(dst_store)
        dst[:] = src.lazy[:]
        np.testing.assert_array_equal(dst[:], data)

    def test_slice_copy(self, tmp_path):
        data = np.arange(100, dtype="int32").reshape(10, 10)
        src_store = LocalStore(str(tmp_path / "src.zarr"))
        dst_store = LocalStore(str(tmp_path / "dst.zarr"))
        save_array(src_store, data)
        save_array(dst_store, np.zeros_like(data))
        src = open_zarrs(src_store)
        dst = open_zarrs(dst_store)
        dst[2:5, 3:7] = src.lazy[2:5, 3:7]
        expected = np.zeros_like(data)
        expected[2:5, 3:7] = data[2:5, 3:7]
        np.testing.assert_array_equal(dst[:], expected)

    def test_different_regions(self, tmp_path):
        data = np.arange(100, dtype="float32").reshape(10, 10)
        src_store = LocalStore(str(tmp_path / "src.zarr"))
        dst_store = LocalStore(str(tmp_path / "dst.zarr"))
        save_array(src_store, data)
        save_array(dst_store, np.zeros_like(data))
        src = open_zarrs(src_store)
        dst = open_zarrs(dst_store)
        dst[0:3, 0:4] = src.lazy[5:8, 6:10]
        expected = np.zeros_like(data)
        expected[0:3, 0:4] = data[5:8, 6:10]
        np.testing.assert_array_equal(dst[:], expected)

    def test_across_chunks(self, tmp_path):
        data = np.arange(100, dtype="int32").reshape(10, 10)
        src_store = LocalStore(str(tmp_path / "src.zarr"))
        dst_store = LocalStore(str(tmp_path / "dst.zarr"))
        save_array(src_store, data, chunks=(3, 3))
        save_array(dst_store, np.zeros_like(data), chunks=(4, 4))
        src = open_zarrs(src_store)
        dst = open_zarrs(dst_store)
        dst[1:8, 2:9] = src.lazy[1:8, 2:9]
        expected = np.zeros_like(data)
        expected[1:8, 2:9] = data[1:8, 2:9]
        np.testing.assert_array_equal(dst[:], expected)

    def test_integer_index(self, tmp_path):
        data = np.arange(30, dtype="float64").reshape(5, 6)
        src_store = LocalStore(str(tmp_path / "src.zarr"))
        dst_store = LocalStore(str(tmp_path / "dst.zarr"))
        save_array(src_store, data)
        save_array(dst_store, np.zeros_like(data))
        src = open_zarrs(src_store)
        dst = open_zarrs(dst_store)
        dst[2] = src.lazy[3]
        expected = np.zeros_like(data)
        expected[2] = data[3]
        np.testing.assert_array_equal(dst[:], expected)

    def test_shape_mismatch(self, tmp_path):
        src_store = LocalStore(str(tmp_path / "src.zarr"))
        dst_store = LocalStore(str(tmp_path / "dst.zarr"))
        save_array(src_store, np.zeros((10, 10), dtype="float32"))
        save_array(dst_store, np.zeros((10, 10), dtype="float32"))
        src = open_zarrs(src_store)
        dst = open_zarrs(dst_store)
        with pytest.raises(ValueError, match="could not broadcast"):
            dst[0:2, 0:3] = src.lazy[0:3, 0:2]

    def test_empty_region(self, tmp_path):
        src_store = LocalStore(str(tmp_path / "src.zarr"))
        dst_store = LocalStore(str(tmp_path / "dst.zarr"))
        save_array(src_store, np.zeros(10, dtype="int64"))
        save_array(dst_store, np.zeros(10, dtype="int64"))
        src = open_zarrs(src_store)
        dst = open_zarrs(dst_store)
        dst[3:3] = src.lazy[5:5]  # empty region, should be a no-op

    def test_np_asarray_full(self, store):
        data = np.arange(20, dtype="float64").reshape(4, 5)
        save_array(store, data)
        arr = open_zarrs(store)
        result = np.asarray(arr.lazy[:])
        np.testing.assert_array_equal(result, data)

    def test_np_asarray_slice(self, store):
        data = np.arange(100, dtype="int32").reshape(10, 10)
        save_array(store, data)
        arr = open_zarrs(store)
        result = np.asarray(arr.lazy[2:5, 3:7])
        np.testing.assert_array_equal(result, data[2:5, 3:7])

    def test_np_asarray_integer_index(self, store):
        data = np.arange(30, dtype="float64").reshape(5, 6)
        save_array(store, data)
        arr = open_zarrs(store)
        result = np.asarray(arr.lazy[2])
        np.testing.assert_array_equal(result, data[2])
        assert result.shape == (6,)

    def test_np_asarray_dtype_cast(self, store):
        data = np.arange(10, dtype="int32")
        save_array(store, data)
        arr = open_zarrs(store)
        result = np.asarray(arr.lazy[:], dtype="float64")
        np.testing.assert_array_equal(result, data.astype("float64"))
        assert result.dtype == np.float64


class TestSubclass:
    def test_isinstance(self, store):
        save_array(store, np.zeros((3, 4), dtype="float32"))
        arr = open_zarrs(store)
        assert isinstance(arr, zarr.Array)

    def test_from_zarr_array(self, store):
        data = np.arange(20, dtype="float64").reshape(4, 5)
        save_array(store, data)
        zarr_arr = zarr.open_array(store)
        arr = ZarrsArray(zarr_arr)
        np.testing.assert_array_equal(arr[:], data)
        assert arr.shape == (4, 5)

    def test_zarr_properties(self, store):
        save_array(store, np.zeros((3, 4), dtype="float32"))
        arr = open_zarrs(store)
        assert arr.store_path is not None
        assert arr.metadata is not None


class TestEllipsis:
    def test_ellipsis_full(self, store):
        data = np.arange(12, dtype="float32").reshape(3, 4)
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[...], data)

    def test_ellipsis_leading(self, store):
        data = np.arange(60, dtype="float64").reshape(3, 4, 5)
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[..., 1:3], data[..., 1:3])

    def test_ellipsis_trailing(self, store):
        data = np.arange(60, dtype="float64").reshape(3, 4, 5)
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[1, ...], data[1, ...])

    def test_ellipsis_middle(self, store):
        data = np.arange(60, dtype="float64").reshape(3, 4, 5)
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[1, ..., 2:4], data[1, ..., 2:4])


class TestFallback:
    def test_step_slicing(self, store):
        data = np.arange(10, dtype="int32")
        save_array(store, data)
        arr = open_zarrs(store)
        np.testing.assert_array_equal(arr[::2], data[::2])

    def test_fancy_indexing(self, store):
        data = np.arange(10, dtype="int32")
        save_array(store, data)
        arr = open_zarrs(store)
        indices = np.array([1, 3, 5])
        np.testing.assert_array_equal(arr[indices], data[indices])


class TestStrictMode:
    def test_strict_rejects_step(self, store):
        save_array(store, np.zeros(10, dtype="float32"))
        arr = open_zarrs(store)
        with (
            zarr.config.set({"codec_pipeline.strict": True}),
            pytest.raises(IndexError, match="advanced indexing"),
        ):
            arr[::2]

    def test_strict_rejects_fancy(self, store):
        save_array(store, np.zeros(10, dtype="float32"))
        arr = open_zarrs(store)
        with (
            zarr.config.set({"codec_pipeline.strict": True}),
            pytest.raises(IndexError, match="advanced indexing"),
        ):
            arr[np.array([1, 2])]

    def test_strict_allows_basic(self, store):
        data = np.arange(10, dtype="int32")
        save_array(store, data)
        arr = open_zarrs(store)
        with zarr.config.set({"codec_pipeline.strict": True}):
            np.testing.assert_array_equal(arr[2:5], data[2:5])

    def test_strict_allows_ellipsis(self, store):
        data = np.arange(12, dtype="float32").reshape(3, 4)
        save_array(store, data)
        arr = open_zarrs(store)
        with zarr.config.set({"codec_pipeline.strict": True}):
            np.testing.assert_array_equal(arr[...], data)
