from __future__ import annotations

import operator
from functools import reduce
from itertools import product
from types import EllipsisType
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import pytest
import zarr
import zarr.codecs
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from _pytest.mark.structures import ParameterSet

    Index: TypeAlias = tuple[int | slice | np.ndarray | EllipsisType, ...]


axis_size_ = 10
chunk_size_ = axis_size_ // 2
fill_value_ = 32767
dimensionalities_ = list(range(1, 5))


@pytest.fixture
def axis_size() -> int:
    return axis_size_


@pytest.fixture
def fill_value() -> int:
    return fill_value_


non_numpy_indices = [
    pytest.param(slice(1, 3), id="slice_in_chunk"),
    pytest.param(slice(1, 7), id="slice_across_chunks"),
    pytest.param(2, id="int"),
    pytest.param(slice(None), id="full_slice"),
    pytest.param(Ellipsis, id="ellipsis"),
]

numpy_indices = [
    pytest.param(np.array([1, 2]), id="contiguous_in_chunk_array"),
    pytest.param(np.array([0, 3]), id="discontinuous_in_chunk_array"),
    pytest.param(np.array([0, 6]), id="across_chunks_indices_array"),
]

all_indices = numpy_indices + non_numpy_indices

indexing_method_params = [
    pytest.param(lambda x: getattr(x, "oindex"), id="oindex"),
    pytest.param(lambda x: x, id="vindex"),
]

zarr_formats = [2, 3]


def _full_array(shape: tuple[int, ...]) -> np.ndarray:
    return np.arange(reduce(operator.mul, shape, 1)).reshape(shape)


@pytest.fixture
def full_array() -> Callable[[tuple[int, ...]], np.ndarray]:
    return _full_array


def gen_arr(
    tmp_path: Path, fill_value: int, dimensionality: int, format: Literal[2, 3]
) -> zarr.Array:
    return zarr.create(
        (axis_size_,) * dimensionality,
        store=LocalStore(root=tmp_path / ".zarr"),
        chunks=(chunk_size_,) * dimensionality,
        dtype=np.int16,
        fill_value=fill_value,
        codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()]
        if format == 3
        else None,
        zarr_format=format,
    )


@pytest.fixture(params=dimensionalities_)
def dimensionality(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=zarr_formats)
def format(request: pytest.FixtureRequest) -> Literal[2, 3]:
    return request.param


@pytest.fixture
def arr(tmp_path: Path, dimensionality: int, format: Literal[2, 3]) -> zarr.Array:
    return gen_arr(tmp_path, fill_value_, dimensionality, format)


# this parameter set is only used for test_roundtrip, but it’s easier to define here


def roundtrip_params() -> Generator[ParameterSet]:
    for format, dimensionality in product(zarr_formats, dimensionalities_):
        indexers = non_numpy_indices if dimensionality > 2 else all_indices
        for index_param_prod in product(indexers, repeat=dimensionality):
            index = tuple(index_param.values[0] for index_param in index_param_prod)
            # multi-ellipsis indexing is not supported
            if sum(isinstance(i, EllipsisType) for i in index) > 1:
                continue
            for indexing_method_param in indexing_method_params:
                id = "-".join(
                    [
                        str(indexing_method_param.id),
                        f"{dimensionality}d",
                        *(str(index_param.id) for index_param in index_param_prod),
                        f"v{format}",
                    ]
                )
                indexing_method = indexing_method_param.values[0]
                yield pytest.param(
                    (format, dimensionality, index, indexing_method), id=id
                )


@pytest.fixture(params=list(roundtrip_params()))
def roundtrip(
    request: pytest.FixtureRequest,
) -> tuple[Literal[2, 3], int, Index, Callable]:
    return request.param
