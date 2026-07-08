from typing import Literal

import numpy as np
import pytest
from zarr.abc.store import Store
from zarr.api.asynchronous import create_array as create_async_array
from zarr.codecs import BytesCodec
from zarr.core.chunk_key_encodings import ChunkKeyEncodingParams
from zarr.storage import StorePath

from .test_codecs import _AsyncArrayProxy


@pytest.mark.parametrize("endian", ["big", "little"])
async def test_endian(store: Store, endian: Literal["big", "little"]) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))
    path = "endian"
    spath = StorePath(store, path)
    a = await create_async_array(
        spath,
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        chunk_key_encoding=ChunkKeyEncodingParams(name="v2", separator="."),
        serializer=BytesCodec(endian=endian),
    )

    await _AsyncArrayProxy(a)[:, :].set(data)
    readback_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, readback_data)


@pytest.mark.parametrize("dtype_input_endian", [">u2", "<u2"])
@pytest.mark.parametrize("dtype_store_endian", ["big", "little"])
async def test_endian_write(
    store: Store,
    dtype_input_endian: Literal[">u2", "<u2"],
    dtype_store_endian: Literal["big", "little"],
) -> None:
    data = np.arange(0, 256, dtype=dtype_input_endian).reshape((16, 16))
    path = "endian"
    spath = StorePath(store, path)
    a = await create_async_array(
        spath,
        shape=data.shape,
        chunks=(16, 16),
        dtype="uint16",
        fill_value=0,
        chunk_key_encoding=ChunkKeyEncodingParams(name="v2", separator="."),
        serializer=BytesCodec(endian=dtype_store_endian),
    )

    await _AsyncArrayProxy(a)[:, :].set(data)
    readback_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, readback_data)
