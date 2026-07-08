import inspect
import warnings
from importlib.metadata import version
from typing import Any, get_args

import numpy as np
import numpy.typing as npt
import pytest
from packaging.version import Version
from zarr import config, create_array
from zarr.abc.store import Store
from zarr.api.asynchronous import create_array as create_async_array
from zarr.codecs import (
    BloscCodec,
    ShardingCodec,
    ShardingCodecIndexLocation,
    TransposeCodec,
)
from zarr.core.array import ShardsConfigParam
from zarr.core.buffer import default_buffer_prototype
from zarr.errors import ZarrUserWarning
from zarr.storage import LocalStore, StorePath

from zarrs.pipeline import SubchunkWriteOrder

from .conftest import ArrayRequest
from .test_codecs import _AsyncArrayProxy, order_from_dim


@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize(
    "array_fixture",
    [
        ArrayRequest(shape=(128,) * 1, dtype="uint8", order="C"),
        ArrayRequest(shape=(128,) * 2, dtype="uint8", order="C"),
        ArrayRequest(shape=(128,) * 3, dtype="uint16", order="F"),
    ],
    indirect=["array_fixture"],
)
@pytest.mark.parametrize("offset", [0, 10])
def test_sharding(
    store: Store,
    array_fixture: npt.NDArray[Any],
    index_location: ShardingCodecIndexLocation,
    offset: int,
) -> None:
    """
    Test that we can create an array with a sharding codec, write data to that array, and get
    the same data out via indexing.
    """
    data = array_fixture
    spath = StorePath(store)
    arr = create_array(
        spath,
        shape=tuple(s + offset for s in data.shape),
        chunks=(32,) * data.ndim,
        shards=ShardsConfigParam(
            shape=(64,) * data.ndim, index_location=index_location
        ),
        dtype=data.dtype,
        fill_value=6,
        filters=[TransposeCodec(order=order_from_dim("F", data.ndim))],
        compressors=[BloscCodec(cname="lz4")],
    )
    write_region = tuple(slice(offset, None) for dim in range(data.ndim))
    arr[write_region] = data

    if offset > 0:
        empty_region = tuple(slice(0, offset) for dim in range(data.ndim))
        assert np.all(arr[empty_region] == arr.metadata.fill_value)

    read_data = arr[write_region]
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize(
    "array_fixture",
    [
        ArrayRequest(shape=(128,) * 3, dtype="uint16", order="F"),
    ],
    indirect=["array_fixture"],
)
def test_sharding_partial(
    store: Store,
    array_fixture: npt.NDArray[Any],
    index_location: ShardingCodecIndexLocation,
) -> None:
    data = array_fixture
    spath = StorePath(store)
    a = create_array(
        spath,
        shape=tuple(a + 10 for a in data.shape),
        chunks=(32, 32, 32),
        shards=ShardsConfigParam(shape=(64, 64, 64), index_location=index_location),
        dtype=data.dtype,
        fill_value=0,
        filters=[TransposeCodec(order=order_from_dim("F", data.ndim))],
        compressors=[BloscCodec(cname="lz4")],
    )

    a[10:, 10:, 10:] = data

    read_data = a[0:10, 0:10, 0:10]
    assert np.all(read_data == 0)

    read_data = a[10:, 10:, 10:]
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize(
    "array_fixture",
    [
        ArrayRequest(shape=(128,) * 3, dtype="uint16", order="F"),
    ],
    indirect=["array_fixture"],
)
def test_sharding_partial_readwrite(
    store: Store,
    array_fixture: npt.NDArray[Any],
    index_location: ShardingCodecIndexLocation,
) -> None:
    data = array_fixture
    spath = StorePath(store)
    a = create_array(
        spath,
        shape=data.shape,
        chunks=(1, data.shape[1], data.shape[2]),
        shards=ShardsConfigParam(shape=data.shape, index_location=index_location),
        dtype=data.dtype,
        fill_value=0,
    )

    a[:] = data

    for axis in range(len(data.shape)):
        for x in range(data.shape[0]):
            selector = [slice(None), slice(None), slice(None)]
            selector[axis] = x
            read_data = a[*tuple(selector)]
            assert np.array_equal(data[*tuple(selector)], read_data)


@pytest.mark.parametrize(
    "array_fixture",
    [
        ArrayRequest(shape=(128,) * 3, dtype="uint16", order="F"),
    ],
    indirect=["array_fixture"],
)
@pytest.mark.parametrize("index_location", ["start", "end"])
def test_sharding_partial_read(
    store: Store,
    array_fixture: npt.NDArray[Any],
    index_location: ShardingCodecIndexLocation,
) -> None:
    data = array_fixture
    spath = StorePath(store)
    a = create_array(
        spath,
        shape=tuple(a + 10 for a in data.shape),
        chunks=(32, 32, 32),
        shards=ShardsConfigParam(shape=(64, 64, 64), index_location=index_location),
        dtype=data.dtype,
        fill_value=1,
        filters=[TransposeCodec(order=order_from_dim("F", data.ndim))],
        compressors=[BloscCodec(cname="lz4")],
    )

    read_data = a[0:10, 0:10, 0:10]
    assert np.all(read_data == 1)


@pytest.mark.parametrize(
    "array_fixture",
    [
        ArrayRequest(shape=(128,) * 3, dtype="uint16", order="F"),
    ],
    indirect=["array_fixture"],
)
@pytest.mark.parametrize("index_location", ["start", "end"])
def test_sharding_partial_overwrite(
    store: Store,
    array_fixture: npt.NDArray[Any],
    index_location: ShardingCodecIndexLocation,
) -> None:
    data = array_fixture[:10, :10, :10]
    spath = StorePath(store)
    a = create_array(
        spath,
        shape=tuple(a + 10 for a in data.shape),
        chunks=(32, 32, 32),
        shards=ShardsConfigParam(shape=(64, 64, 64), index_location=index_location),
        dtype=data.dtype,
        fill_value=1,
        filters=[TransposeCodec(order=order_from_dim("F", data.ndim))],
        compressors=[BloscCodec(cname="lz4")],
    )

    a[:10, :10, :10] = data

    read_data = a[0:10, 0:10, 0:10]
    assert np.array_equal(data, read_data)

    data = data + 10
    a[:10, :10, :10] = data
    read_data = a[0:10, 0:10, 0:10]
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize(
    "array_fixture",
    [
        ArrayRequest(shape=(128,) * 3, dtype="uint16", order="F"),
    ],
    indirect=["array_fixture"],
)
@pytest.mark.parametrize("outer_index_location", ["start", "end"])
@pytest.mark.parametrize("inner_index_location", ["start", "end"])
def test_nested_sharding(
    store: Store,
    array_fixture: npt.NDArray[Any],
    outer_index_location: ShardingCodecIndexLocation,
    inner_index_location: ShardingCodecIndexLocation,
) -> None:
    data = array_fixture
    spath = StorePath(store)
    warnings.filterwarnings("ignore", r".*performance", ZarrUserWarning)
    a = create_array(
        spath,
        shape=data.shape,
        chunks=(64, 64, 64),
        dtype=data.dtype,
        fill_value=0,
        serializer=ShardingCodec(
            chunk_shape=(32, 32, 32),
            index_location=outer_index_location,
            codecs=[
                ShardingCodec(
                    chunk_shape=(16, 16, 16), index_location=inner_index_location
                )
            ],
        ),
    )

    a[:, :, :] = data

    read_data = a[0 : data.shape[0], 0 : data.shape[1], 0 : data.shape[2]]
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


def test_write_partial_sharded_chunks(store: Store) -> None:
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))
    spath = StorePath(store)
    a = create_array(
        spath,
        shape=(40, 40),
        chunks=(10, 10),
        shards=(20, 20),
        dtype=data.dtype,
        fill_value=1,
        compressors=[BloscCodec()],
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


async def test_delete_empty_shards(store: Store) -> None:
    if not store.supports_deletes:
        pytest.skip("store does not support deletes")
    path = "delete_empty_shards"
    spath = StorePath(store, path)
    a = await create_async_array(
        spath,
        shape=(16, 16),
        chunks=(8, 8),
        shards=(8, 16),
        dtype="uint16",
        fill_value=1,
        compressors=[],
    )
    await _AsyncArrayProxy(a)[:, :].set(np.zeros((16, 16)))
    await _AsyncArrayProxy(a)[8:, :].set(np.ones((8, 16)))
    await _AsyncArrayProxy(a)[:, 8:].set(np.ones((16, 8)))
    # chunk (0, 0) is full
    # chunks (0, 1), (1, 0), (1, 1) are empty
    # shard (0, 0) is half-full
    # shard (1, 0) is empty

    data = np.ones((16, 16), dtype="uint16")
    data[:8, :8] = 0
    assert np.array_equal(data, await _AsyncArrayProxy(a)[:, :].get())
    assert (
        await store.get(f"{path}/c/1/0", prototype=default_buffer_prototype()) is None
    )
    chunk_bytes = await store.get(f"{path}/c/0/0", prototype=default_buffer_prototype())
    assert chunk_bytes is not None
    assert len(chunk_bytes) == 16 * 2 + 8 * 8 * 2 + 4 == 164


@pytest.mark.parametrize(
    "index_location", [ShardingCodecIndexLocation.start, ShardingCodecIndexLocation.end]
)
async def test_sharding_with_empty_inner_chunk(
    store: Store, index_location: ShardingCodecIndexLocation
) -> None:
    data = np.arange(0, 16 * 16, dtype="uint32").reshape((16, 16))
    fill_value = 1

    path = f"sharding_with_empty_inner_chunk_{index_location}"
    spath = StorePath(store, path)
    a = await create_async_array(
        spath,
        shape=(16, 16),
        chunks=(4, 4),
        shards=ShardsConfigParam(shape=(8, 8), index_location=index_location),
        dtype="uint32",
        fill_value=fill_value,
    )
    data[:4, :4] = fill_value
    await a.setitem(..., data)
    print("read data")
    data_read = await a.getitem(...)
    assert np.array_equal(data_read, data)


_SHARDING_HAS_WRITE_ORDER = (
    "subchunk_write_order" in inspect.signature(ShardingCodec).parameters
)

# `subchunk_write_order` on the sharding codec is a zarr-python >=3.2.2 feature.
requires_write_order = pytest.mark.skipif(
    Version(version("zarr")) < Version("3.2.2dev0"),
    reason="zarr-python ShardingCodec has no subchunk_write_order",
)


@requires_write_order
@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("subchunk_write_order", list(get_args(SubchunkWriteOrder)))
def test_subchunk_write_order_matches_zarr_python(
    tmp_path, *, subchunk_write_order: SubchunkWriteOrder, nested: bool
) -> None:
    data = np.arange(1, 32 * 32 + 1, dtype="uint32").reshape((32, 32))
    ground_truth_subchunk_write_order = (
        "unordered"
        if subchunk_write_order in {"colexicographic", "unordered", "morton"}
        else "lexicographic"
    )
    if nested:
        zarrs_codec = ShardingCodec(
            chunk_shape=(8, 8),
            subchunk_write_order=subchunk_write_order,
            codecs=[
                ShardingCodec(chunk_shape=(2, 2), subchunk_write_order="lexicographic")
            ],
        )
        zarr_codec = ShardingCodec(
            chunk_shape=(8, 8),
            subchunk_write_order=ground_truth_subchunk_write_order,
            codecs=[
                ShardingCodec(chunk_shape=(2, 2), subchunk_write_order="lexicographic")
            ],
        )
    else:
        zarrs_codec = ShardingCodec(
            chunk_shape=(8, 8), subchunk_write_order="lexicographic"
        )
        zarr_codec = ShardingCodec(
            chunk_shape=(8, 8), subchunk_write_order=ground_truth_subchunk_write_order
        )

    def write(pipeline: str) -> bytes:
        sub = tmp_path / pipeline.rsplit(".", 1)[-1]
        with config.set({"codec_pipeline.path": pipeline}):
            a = create_array(
                StorePath(LocalStore(sub)),
                shape=(32, 32),
                chunks=(32, 32),
                dtype="uint32",
                fill_value=0,
                serializer=zarrs_codec if "zarrs" in pipeline else zarr_codec,
                compressors=None,
            )
            a[:, :] = data
        return (sub / "c" / "0" / "0").read_bytes()

    zarrs_bytes = write("zarrs.ZarrsCodecPipeline")
    zarr_bytes = write("zarr.core.codec_pipeline.BatchedCodecPipeline")
    assert zarrs_bytes == zarr_bytes
