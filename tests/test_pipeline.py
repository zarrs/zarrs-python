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


@pytest.fixture
def fill_value() -> int:
    return 32767


@pytest.fixture
def chunks() -> tuple[int, ...]:
    return (5, 5)


@pytest.fixture
def shape() -> tuple[int, ...]:
    return (10, 10)


@pytest.fixture(
    params=[
        np.array([1, 2]),
        np.array([0, 3]),
        slice(1, 3),
        slice(1, 7),
        np.array([0, 6]),
        slice(None),
        2,
        Ellipsis,
    ],
    ids=[
        "contiguous_in_chunk_array",
        "discontinuous_in_chunk_array",
        "slice_in_chunk",
        "slice_across_chunks",
        "across_chunks_indices_array",
        "fill_slice",
        "int",
        "ellipsis",
    ],
)
def indexer_1d(request) -> slice | np.ndarray | int | EllipsisType:
    return request.param


@pytest.fixture
def full_array(shape) -> np.ndarray:
    return np.arange(reduce(operator.mul, shape, 1)).reshape(shape)


indexer_1d_2 = indexer_1d


@pytest.fixture
def index(indexer_1d, indexer_1d_2):
    if isinstance(indexer_1d, EllipsisType) and isinstance(indexer_1d_2, EllipsisType):
        pytest.skip("Double ellipsis indexing is valid")
    return indexer_1d, indexer_1d_2


@pytest.fixture(
    params=[lambda x: getattr(x, "oindex"), lambda x: x], ids=["oindex", "vindex"]
)
def indexing_method(request) -> Callable:
    return request.param


@pytest.fixture
def store_values(
    indexing_method: Callable,
    index: tuple[int | slice | np.ndarray | EllipsisType, ...],
    full_array: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    class smoke:
        oindex = None

    if not isinstance(index, EllipsisType) and indexing_method(smoke) == "oindex":
        index: tuple[int | np.ndarray, ...] = tuple(
            i
            if (not isinstance(i, slice))
            else np.arange(
                i.start if hasattr(i, "start") else 0,
                i.stop if hasattr(i, "end") else shape[axis],
            )
            for axis, i in enumerate(index)
        )
        return full_array[np.ix_(index)]
    return full_array[index]


@pytest.fixture
def arr(fill_value, chunks, shape, tmp_path) -> zarr.Array:
    return zarr.create(
        shape,
        store=LocalStore(root=tmp_path / ".zarr", mode="w"),
        chunks=chunks,
        dtype=np.int16,
        fill_value=fill_value,
        codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()],
    )


def test_fill_value(arr: zarr.Array, fill_value: int):
    assert np.all(arr[:] == fill_value)


def test_roundtrip_constant(arr: zarr.Array):
    arr[:] = 42
    assert np.all(arr[:] == 42)


def test_roundtrip_singleton(arr: zarr.Array):
    arr[1, 1] = 42
    assert arr[1, 1] == 42
    assert arr[0, 0] != 42


def test_roundtrip_full_array(arr: zarr.Array, shape: tuple[int, ...]):
    stored_values = np.arange(reduce(operator.mul, shape, 1)).reshape(shape)
    arr[:] = stored_values
    assert np.all(arr[:] == stored_values)


def test_roundtrip(
    arr: zarr.Array,
    store_values: np.ndarray,
    index: tuple[int | slice | np.ndarray | EllipsisType, ...],
    indexing_method: Callable,
):
    if not isinstance(index, EllipsisType) and all(
        isinstance(i, np.ndarray) for i in index
    ):
        pytest.skip(
            "indexing across two axes with arrays seems to have strange behavior even in normal zarr"
        )
    indexing_method(arr)[index] = store_values
    res = indexing_method(arr)[index]
    assert np.all(
        res == store_values,
    ), res


@contextmanager
def use_zarr_default_codec_reader():
    zarr.config.set(
        {"codec_pipeline.path": "zarr.codecs.pipeline.BatchedCodecPipeline"}
    )
    yield
    zarr.config.set({"codec_pipeline.path": "zarrs_python.ZarrsCodecPipeline"})


def test_roundtrip_read_only_zarrs(
    arr,
    store_values: np.ndarray,
    index: tuple[int | slice | np.ndarray | EllipsisType, ...],
    indexing_method: Callable,
):
    if not isinstance(index, EllipsisType) and all(
        isinstance(i, np.ndarray) for i in index
    ):
        pytest.skip(
            "indexing across two axes with arrays seems to have strange behavior even in normal zarr"
        )
    with use_zarr_default_codec_reader():
        arr_default = zarr.open(arr.store, mode="r+")
        indexing_method(arr_default)[index] = store_values
    res = indexing_method(arr)[index]
    assert np.all(
        res == store_values,
    ), res


def test_roundtrip_ellipsis_indexing_1d_invalid(arr: zarr.Array):
    stored_value = np.array([1, 2, 3])
    with pytest.raises(
        BaseException  # TODO: ValueError, but this raises pyo3_runtime.PanicException  # noqa: PT011
    ):
        # zarrs-python error: ValueError: operands could not be broadcast together with shapes (4,) (3,)
        # numpy error: ValueError: could not broadcast input array from shape (3,) into shape (4,)
        arr[2, ...] = stored_value
