from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from types import EllipsisType
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    Index: TypeAlias = tuple[int | slice | np.ndarray | EllipsisType, ...]


@pytest.fixture
def store_values(
    roundtrip: tuple[Literal[2, 3], int, Index, Callable],
    axis_size: int,
    full_array: Callable[[tuple[int, ...]], np.ndarray],
) -> np.ndarray:
    _, dimensionality, index, indexing_method = roundtrip
    return gen_store_values(
        indexing_method,
        index,
        full_array((axis_size,) * dimensionality),
    )


def gen_store_values(
    indexing_method: Callable, index: Index, full_array: np.ndarray
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
                i.stop if i.stop is not None else full_array.shape[axis],
            )
        if isinstance(i, int):
            return np.array([i])
        if isinstance(i, EllipsisType):
            return np.arange(full_array.shape[axis])
        raise ValueError(f"Invalid index {i}")

    if not isinstance(index, EllipsisType) and indexing_method(smoke()) == "oindex":
        index = tuple(maybe_convert(i, axis) for axis, i in enumerate(index))
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


# overwrite format and dimensionality from conftest


@pytest.fixture
def format(roundtrip: tuple[Literal[2, 3], int, Index, Callable]) -> Literal[2, 3]:
    return roundtrip[0]


@pytest.fixture
def dimensionality(roundtrip: tuple[Literal[2, 3], int, Index, Callable]) -> int:
    return roundtrip[1]


@pytest.fixture
def index(roundtrip: tuple[Literal[2, 3], int, Index, Callable]) -> Index:
    return roundtrip[2]


@pytest.fixture
def indexing_method(roundtrip: tuple[Literal[2, 3], int, Index, Callable]) -> Callable:
    return roundtrip[3]


@contextmanager
def use_zarr_default_codec_reader() -> Generator[None]:
    zarr.config.set(
        {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
    )
    yield
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


def test_roundtrip(
    arr: zarr.Array, store_values: np.ndarray, index: Index, indexing_method: Callable
) -> None:
    indexing_method(arr)[index] = store_values
    res = indexing_method(arr)[index]
    assert np.all(res == store_values), res


def test_roundtrip_read_only_zarrs(
    arr: zarr.Array, store_values: np.ndarray, index: Index, indexing_method: Callable
) -> None:
    with use_zarr_default_codec_reader():
        arr_default = zarr.open(arr.store, read_only=True)
        indexing_method(arr_default)[index] = store_values
    res = indexing_method(zarr.open(arr.store))[index]
    assert np.all(
        res == store_values,
    ), res
