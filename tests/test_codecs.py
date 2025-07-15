from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
from zarr import Array, AsyncArray, config
from zarr.codecs import (
    BytesCodec,
    ShardingCodec,
    TransposeCodec,
)
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import StorePath

if TYPE_CHECKING:
    from zarr.abc.codec import Codec
    from zarr.abc.store import Store
    from zarr.core.buffer.core import NDArrayLike
    from zarr.core.common import MemoryOrder
    from zarr.core.indexing import Selection


@dataclass(frozen=True)
class _AsyncArrayProxy:
    array: AsyncArray

    def __getitem__(self, selection: Selection) -> _AsyncArraySelectionProxy:
        return _AsyncArraySelectionProxy(self.array, selection)


@dataclass(frozen=True)
class _AsyncArraySelectionProxy:
    array: AsyncArray
    selection: Selection

    async def get(self) -> NDArrayLike:
        return await self.array.getitem(self.selection)

    async def set(self, value: np.ndarray) -> None:
        return await self.array.setitem(self.selection, value)


def order_from_dim(order: MemoryOrder, ndim: int) -> tuple[int, ...]:
    if order == "F":
        return tuple(ndim - x - 1 for x in range(ndim))
    else:
        return tuple(range(ndim))


def test_sharding_pickle() -> None:
    """
    Test that sharding codecs can be pickled
    """
    pass


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("store_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["C"])
@pytest.mark.parametrize("runtime_read_order", ["C"])
@pytest.mark.parametrize("with_sharding", [True, False])
async def test_order(
    *,
    store: Store,
    input_order: MemoryOrder,
    store_order: MemoryOrder,
    runtime_write_order: MemoryOrder,
    runtime_read_order: MemoryOrder,
    with_sharding: bool,
) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((32, 8), order=input_order)
    path = "order"
    spath = StorePath(store, path=path)
    codecs_: list[Codec] = (
        [
            ShardingCodec(
                chunk_shape=(16, 8),
                codecs=[
                    TransposeCodec(order=order_from_dim(store_order, data.ndim)),
                    BytesCodec(),
                ],
            )
        ]
        if with_sharding
        else [
            TransposeCodec(order=order_from_dim(store_order, data.ndim)),
            BytesCodec(),
        ]
    )

    with config.set({"array.order": runtime_write_order}):
        a = await AsyncArray.create(
            spath,
            shape=data.shape,
            chunk_shape=(32, 8),
            dtype=data.dtype,
            fill_value=0,
            chunk_key_encoding=("v2", "."),
            codecs=codecs_,
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


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["C"])
@pytest.mark.parametrize("runtime_read_order", ["C"])
@pytest.mark.parametrize("with_sharding", [True, False])
def test_order_implicit(
    *,
    store: Store,
    input_order: MemoryOrder,
    runtime_write_order: MemoryOrder,
    runtime_read_order: MemoryOrder,
    with_sharding: bool,
) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order=input_order)
    path = "order_implicit"
    spath = StorePath(store, path)
    codecs_: list[Codec] | None = (
        [ShardingCodec(chunk_shape=(8, 8))] if with_sharding else None
    )

    with config.set({"array.order": runtime_write_order}):
        a = Array.create(
            spath,
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=codecs_,
        )

    a[:, :] = data

    with config.set({"array.order": runtime_read_order}):
        a = Array.open(spath)
    read_data = a[:, :]
    assert np.array_equal(data, read_data)

    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]


def test_write_partial_chunks(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))
    spath = StorePath(store)
    a = Array.create(
        spath,
        shape=data.shape,
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


async def test_delete_empty_chunks(store: Store) -> None:
    data = np.ones((16, 16))
    path = "delete_empty_chunks"
    spath = StorePath(store, path)
    a = await AsyncArray.create(
        spath,
        shape=data.shape,
        chunk_shape=(32, 32),
        dtype=data.dtype,
        fill_value=1,
    )
    await _AsyncArrayProxy(a)[:16, :16].set(np.zeros((16, 16)))
    await _AsyncArrayProxy(a)[:16, :16].set(data)
    assert np.array_equal(await _AsyncArrayProxy(a)[:16, :16].get(), data)
    assert await store.get(f"{path}/c0/0", prototype=default_buffer_prototype()) is None


def test_invalid_metadata(store: Store) -> None:
    # LD: Disabled for `zarrs`. Including endianness for a single-byte data type is not invalid.
    # spath2 = StorePath(store, "invalid_endian")
    # with pytest.raises(TypeError):
    #     Array.create(
    #         spath2,
    #         shape=(16, 16),
    #         chunk_shape=(16, 16),
    #         dtype=np.dtype("uint8"),
    #         fill_value=0,
    #         codecs=[
    #             BytesCodec(endian="big"),
    #             TransposeCodec(order=order_from_dim("F", 2)),
    #         ],
    #     )
    spath3 = StorePath(store, "invalid_order")
    with pytest.raises(TypeError):
        Array.create(
            spath3,
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                BytesCodec(),
                TransposeCodec(order="F"),  # type: ignore[arg-type]
            ],
        )
    spath4 = StorePath(store, "invalid_missing_bytes_codec")
    with pytest.raises(ValueError, match=r".*[Cc]odec.*required"):
        Array.create(
            spath4,
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                TransposeCodec(order=order_from_dim("F", 2)),
            ],
        )
    spath5 = StorePath(store, "invalid_inner_chunk_shape")
    with pytest.raises(
        ValueError, match=r".*shard.*chunk_shape.*array.*shape.*need.*same.*dimensions"
    ):
        Array.create(
            spath5,
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                ShardingCodec(chunk_shape=(8,)),
            ],
        )
    spath6 = StorePath(store, "invalid_inner_chunk_shape")
    with pytest.raises(
        ValueError, match=r".*array.*chunk_shape.*divisible.*shard.*chunk_shape"
    ):
        Array.create(
            spath6,
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                ShardingCodec(chunk_shape=(8, 7)),
            ],
        )
    # LD: Disabled for `zarrs`. Such checks do not exist.
    #     Also this is not invalid metadata, should be a separate test.
    # spath7 = StorePath(store, "warning_inefficient_codecs")
    # with pytest.warns(UserWarning):
    #     Array.create(
    #         spath7,
    #         shape=(16, 16),
    #         chunk_shape=(16, 16),
    #         dtype=np.dtype("uint8"),
    #         fill_value=0,
    #         codecs=[
    #             ShardingCodec(chunk_shape=(8, 8)),
    #             GzipCodec(),
    #         ],
    #     )


async def test_resize(store: Store) -> None:
    data = np.zeros((16, 18), dtype="uint16")
    path = "resize"
    spath = StorePath(store, path)
    a = await AsyncArray.create(
        spath,
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding=("v2", "."),
        fill_value=1,
    )

    await _AsyncArrayProxy(a)[:16, :18].set(data)
    assert (
        await store.get(f"{path}/1.1", prototype=default_buffer_prototype()) is not None
    )
    assert (
        await store.get(f"{path}/0.0", prototype=default_buffer_prototype()) is not None
    )
    assert (
        await store.get(f"{path}/0.1", prototype=default_buffer_prototype()) is not None
    )
    assert (
        await store.get(f"{path}/1.0", prototype=default_buffer_prototype()) is not None
    )

    await a.resize((10, 12))
    assert a.metadata.shape == (10, 12)
    assert (
        await store.get(f"{path}/0.0", prototype=default_buffer_prototype()) is not None
    )
    assert (
        await store.get(f"{path}/0.1", prototype=default_buffer_prototype()) is not None
    )
    assert await store.get(f"{path}/1.0", prototype=default_buffer_prototype()) is None
    assert await store.get(f"{path}/1.1", prototype=default_buffer_prototype()) is None
