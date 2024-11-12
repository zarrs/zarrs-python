#!/usr/bin/env python3

import operator
from collections.abc import Callable
from contextlib import contextmanager
from functools import reduce
from types import EllipsisType

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

import zarrs_python  # noqa: F401

axis_size = 10
chunk_size = axis_size // 2


@pytest.fixture
def fill_value() -> int:
    return 32767


non_numpy_indices = [
    pytest.param(slice(1, 3), id="slice_in_chunk"),
    pytest.param(slice(1, 7), id="slice_across_chunks"),
    pytest.param(2, id="int"),
    pytest.param(slice(None), id="full_slice"),
    pytest.param(Ellipsis, id="ellipsis"),
]


@pytest.fixture(
    params=[
        pytest.param(np.array([1, 2]), id="contiguous_in_chunk_array"),
        pytest.param(np.array([0, 3]), id="discontinuous_in_chunk_array"),
        pytest.param(np.array([0, 6]), id="across_chunks_indices_array"),
        *non_numpy_indices,
    ],
)
def indexer_1d_with_numpy(request) -> slice | np.ndarray | int | EllipsisType:
    return request.param


@pytest.fixture(
    params=non_numpy_indices,
)
def indexer_1d_no_numpy(request) -> slice | np.ndarray | int | EllipsisType:
    return request.param


def full_array(shape) -> np.ndarray:
    return np.arange(reduce(operator.mul, shape, 1)).reshape(shape)


@pytest.fixture
def full_array_2d() -> np.ndarray:
    shape = (axis_size,) * 2
    return full_array(shape)


@pytest.fixture
def full_array_3d() -> np.ndarray:
    shape = (axis_size,) * 3
    return full_array(shape)


indexer_1d_with_numpy_2 = indexer_1d_with_numpy
indexer_1d_no_numpy_2 = indexer_1d_no_numpy
indexer_1d_no_numpy_3 = indexer_1d_no_numpy


@pytest.fixture
def index_2d(indexer_1d_with_numpy, indexer_1d_with_numpy_2):
    if isinstance(indexer_1d_with_numpy, EllipsisType) and isinstance(
        indexer_1d_with_numpy_2, EllipsisType
    ):
        pytest.skip("Double ellipsis indexing is not valid")
    return indexer_1d_with_numpy, indexer_1d_with_numpy_2


@pytest.fixture
def index_3d(indexer_1d_no_numpy, indexer_1d_no_numpy_2, indexer_1d_no_numpy_3):
    if (
        sum(
            isinstance(ind, EllipsisType)
            for ind in [
                indexer_1d_no_numpy,
                indexer_1d_no_numpy_2,
                indexer_1d_no_numpy_3,
            ]
        )
        > 1
    ):
        pytest.skip("Multi ellipsis indexing is not valid")
    return indexer_1d_no_numpy, indexer_1d_no_numpy_2, indexer_1d_no_numpy_3


@pytest.fixture(
    params=[lambda x: getattr(x, "oindex"), lambda x: x], ids=["oindex", "vindex"]
)
def indexing_method(request) -> Callable:
    return request.param


@pytest.fixture
def store_values_2d(
    indexing_method: Callable,
    index_2d: tuple[int | slice | np.ndarray | EllipsisType, ...],
    full_array_2d: np.ndarray,
) -> np.ndarray:
    return store_values(indexing_method, index_2d, full_array_2d, (axis_size,) * 2)


@pytest.fixture
def store_values_3d(
    indexing_method: Callable,
    index_3d: tuple[int | slice | np.ndarray | EllipsisType, ...],
    full_array_3d: np.ndarray,
) -> np.ndarray:
    return store_values(indexing_method, index_3d, full_array_3d, (axis_size,) * 3)


def store_values(
    indexing_method: Callable,
    index: tuple[int | slice | np.ndarray | EllipsisType, ...],
    full_array: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    class smoke:
        oindex = "oindex"

    def maybe_convert(
        i: int | np.ndarray | slice | EllipsisType, axis: int
    ) -> np.ndarray:
        if isinstance(i, np.ndarray):
            return i
        if isinstance(i, slice):
            return np.arange(
                i.start if i.start is not None else 0,
                i.stop if i.stop is not None else shape[axis],
            )
        if isinstance(i, int):
            return np.array([i])
        if isinstance(i, EllipsisType):
            return np.arange(shape[axis])
        raise ValueError(f"Invalid index {i}")

    if not isinstance(index, EllipsisType) and indexing_method(smoke()) == "oindex":
        index: tuple[np.ndarray, ...] = tuple(
            maybe_convert(i, axis) for axis, i in enumerate(index)
        )
        res = full_array[np.ix_(*index)]
        # squeeze out extra dims from integer indexers
        if all(i.shape == (1,) for i in index):
            res = res.squeeze()
            return res
        res = res.squeeze(
            axis=tuple(axis for axis, i in enumerate(index) if i.shape == (1,))
        )
        return res
    return full_array[index]


@pytest.fixture
def arr_2d(fill_value, tmp_path) -> zarr.Array:
    return zarr.create(
        (axis_size,) * 2,
        store=LocalStore(root=tmp_path / ".zarr", mode="w"),
        chunks=(chunk_size,) * 2,
        dtype=np.int16,
        fill_value=fill_value,
        codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()],
    )


@pytest.fixture
def arr_3d(fill_value, tmp_path) -> zarr.Array:
    return zarr.create(
        (axis_size,) * 3,
        store=LocalStore(root=tmp_path / ".zarr", mode="w"),
        chunks=(chunk_size,) * 3,
        dtype=np.int16,
        fill_value=fill_value,
        codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()],
    )


def test_fill_value(arr_2d: zarr.Array, fill_value: int):
    assert np.all(arr_2d[:] == fill_value)


def test_roundtrip_constant(arr_2d: zarr.Array):
    arr_2d[:] = 42
    assert np.all(arr_2d[:] == 42)


def test_roundtrip_singleton(arr_2d: zarr.Array):
    arr_2d[1, 1] = 42
    assert arr_2d[1, 1] == 42
    assert arr_2d[0, 0] != 42


def test_roundtrip_full_array(arr_2d: zarr.Array):
    stored_values = np.arange(reduce(operator.mul, arr_2d.shape, 1)).reshape(
        arr_2d.shape
    )
    arr_2d[:] = stored_values
    assert np.all(arr_2d[:] == stored_values)


def test_roundtrip_2d(
    arr_2d: zarr.Array,
    store_values_2d: np.ndarray,
    index_2d: tuple[int | slice | np.ndarray | EllipsisType, ...],
    indexing_method: Callable,
):
    indexing_method(arr_2d)[index_2d] = store_values_2d
    res = indexing_method(arr_2d)[index_2d]
    assert np.all(
        res == store_values_2d,
    ), res


@contextmanager
def use_zarr_default_codec_reader():
    zarr.config.set(
        {"codec_pipeline.path": "zarr.codecs.pipeline.BatchedCodecPipeline"}
    )
    yield
    zarr.config.set({"codec_pipeline.path": "zarrs_python.ZarrsCodecPipeline"})


def test_roundtrip_read_only_zarrs_2d(
    arr_2d: zarr.Array,
    store_values_2d: np.ndarray,
    index_2d: tuple[int | slice | np.ndarray | EllipsisType, ...],
    indexing_method: Callable,
):
    with use_zarr_default_codec_reader():
        arr_default = zarr.open(arr_2d.store, mode="r+")
        indexing_method(arr_default)[index_2d] = store_values_2d
    res = indexing_method(zarr.open(arr_2d.store))[index_2d]
    assert np.all(
        res == store_values_2d,
    ), res


def test_roundtrip_ellipsis_indexing_1d_invalid(arr_2d: zarr.Array):
    stored_value = np.array([1, 2, 3])
    with pytest.raises(
        BaseException  # TODO: ValueError, but this raises pyo3_runtime.PanicException  # noqa: PT011
    ):
        # zarrs-python error: ValueError: operands could not be broadcast together with shapes (4,) (3,)
        # numpy error: ValueError: could not broadcast input array from shape (3,) into shape (4,)
        arr_2d[2, ...] = stored_value


def test_roundtrip_3d(
    arr_3d: zarr.Array,
    store_values_3d: np.ndarray,
    index_3d: tuple[int | slice | np.ndarray | EllipsisType, ...],
    indexing_method: Callable,
):
    indexing_method(arr_3d)[index_3d] = store_values_3d
    res = indexing_method(arr_3d)[index_3d]
    assert np.all(
        res == store_values_3d,
    ), res


def test_roundtrip_read_only_zarrs_3d(
    arr_3d: zarr.Array,
    store_values_3d: np.ndarray,
    index_3d: tuple[int | slice | np.ndarray | EllipsisType, ...],
    indexing_method: Callable,
):
    with use_zarr_default_codec_reader():
        arr_default = zarr.open(arr_3d.store, mode="r+")
        indexing_method(arr_default)[index_3d] = store_values_3d
    res = indexing_method(zarr.open(arr_3d.store))[index_3d]
    assert np.all(
        res == store_values_3d,
    ), res
