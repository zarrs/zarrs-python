from __future__ import annotations

import operator
import pickle
import platform
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import pytest
import zarr
import zarr.abc.codec
import zarr.codecs

import zarrs

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import EllipsisType

    Index: TypeAlias = tuple[int | slice | np.ndarray | EllipsisType, ...]


def test_fill_value(arr: zarr.Array, fill_value: int) -> None:
    assert np.all(arr[:] == fill_value)


def test_constant(arr: zarr.Array):
    arr[:] = 42
    assert np.all(arr[:] == 42)


def test_singleton(arr: zarr.Array):
    singleton_index = (1,) * len(arr.shape)
    non_singleton_index = (0,) * len(arr.shape)
    arr[singleton_index] = 42
    assert arr[singleton_index] == 42
    assert arr[non_singleton_index] != 42


def test_full_array(
    arr: zarr.Array, full_array: Callable[[tuple[int, ...]], np.ndarray]
) -> None:
    stored_values = full_array(arr.shape)
    arr[:] = stored_values
    assert np.all(arr[:] == stored_values)


def test_ellipsis_indexing_invalid(arr: zarr.Array):
    if len(arr.shape) <= 2:
        pytest.skip(
            "Ellipsis indexing works for 1D and 2D arrays in zarr-python despite a shape mismatch"
        )
    stored_value = np.array([1, 2, 3])
    expected_errors = (
        "references array indices.*out-of-bounds of array shape",
        "the size of the chunk subset.*and input/output subset.* are incompatible",
    )
    with pytest.raises(IndexError, match="|".join(expected_errors)):
        arr[2, ...] = stored_value


def test_pickle(arr: zarr.Array, tmp_path: Path):
    arr[:] = np.arange(reduce(operator.mul, arr.shape, 1)).reshape(arr.shape)
    expected = arr[:]
    with Path.open(tmp_path / "arr.pickle", "wb") as f:
        pickle.dump(arr._async_array.codec_pipeline, f)
    with Path.open(tmp_path / "arr.pickle", "rb") as f:
        object.__setattr__(arr._async_array, "codec_pipeline", pickle.load(f))
    assert (arr[:] == expected).all()


@pytest.mark.parametrize(
    "codec",
    [zarr.codecs.BloscCodec(), zarr.codecs.GzipCodec(), zarr.codecs.ZstdCodec()],
)
@pytest.mark.parametrize("should_shard", [True, False])
def test_pipeline_used(
    mocker, codec: zarr.abc.codec.BaseCodec, tmp_path: Path, *, should_shard: bool
):
    z = zarr.create_array(
        tmp_path / "foo.zarr",
        dtype=np.float64,
        shape=(80, 100),
        chunks=(10, 10),
        shards=(20, 20) if should_shard else None,
        compressors=[codec],
    )
    spy_read = mocker.spy(z._async_array.codec_pipeline, "read")
    spy_write = mocker.spy(z._async_array.codec_pipeline, "write")
    assert isinstance(z._async_array.codec_pipeline, zarrs.ZarrsCodecPipeline)
    z[...] = np.random.random(z.shape)
    z[...]
    assert spy_read.call_count == 1
    assert spy_write.call_count == 1


@pytest.fixture
def use_zarrs_direct_io() -> Generator[None]:
    zarr.config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
            "codec_pipeline.direct_io": True,
        }
    )
    yield
    zarr.config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
            "codec_pipeline.direct_io": False,
        }
    )


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Can only run O_DIRECT on linux"
)
@pytest.mark.usefixtures("use_zarrs_direct_io")
def test_direct_io(tmp_path: Path) -> None:
    z = zarr.create_array(
        tmp_path / "foo.zarr",
        dtype=np.float64,
        shape=(80, 100),
        chunks=(10, 10),
        shards=(20, 20),
    )
    ground_truth_arr = np.random.random(z.shape)
    z[...] = ground_truth_arr
    np.testing.assert_array_equal(z[...], ground_truth_arr)
