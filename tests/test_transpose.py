import numpy as np
import pytest
from zarr import AsyncArray, config, create_array
from zarr.abc.store import Store
from zarr.api.asynchronous import create_array as create_async_array
from zarr.codecs import TransposeCodec
from zarr.core.chunk_key_encodings import ChunkKeyEncodingParams
from zarr.core.common import MemoryOrder
from zarr.storage import StorePath

from .test_codecs import _AsyncArrayProxy


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["C"])
@pytest.mark.parametrize("runtime_read_order", ["C"])
@pytest.mark.parametrize("with_sharding", [True, False])
async def test_transpose(
    *,
    store: Store,
    input_order: MemoryOrder,
    runtime_write_order: MemoryOrder,
    runtime_read_order: MemoryOrder,
    with_sharding: bool,
) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((1, 32, 8), order=input_order)
    spath = StorePath(store, path="transpose")
    with config.set({"array.order": runtime_write_order}):
        a = await create_async_array(
            spath,
            shape=data.shape,
            shards=(1, 32, 8) if with_sharding else None,
            dtype=data.dtype,
            fill_value=0,
            chunk_key_encoding=ChunkKeyEncodingParams(name="v2", separator="."),
            filters=[TransposeCodec(order=(2, 1, 0))],
        )

    await _AsyncArrayProxy(a)[:, :].set(data)
    read_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, read_data)

    with config.set({"array.order": runtime_read_order}):
        a = await AsyncArray.open(
            spath,
        )
    read_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, read_data)

    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("order", [[1, 2, 0], [1, 2, 3, 0], [3, 2, 4, 0, 1]])
def test_transpose_non_self_inverse(store: Store, order: list[int]) -> None:
    shape = [i + 3 for i in range(len(order))]
    data = np.arange(0, np.prod(shape), dtype="uint16").reshape(shape)
    spath = StorePath(store, "transpose_non_self_inverse")
    a = create_array(
        spath,
        shape=data.shape,
        chunks=data.shape,
        dtype=data.dtype,
        fill_value=0,
        filters=[TransposeCodec(order=order)],
    )
    a[:, :] = data
    read_data = a[:, :]
    assert np.array_equal(data, read_data)
