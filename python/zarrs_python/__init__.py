from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.abc.codec import (
    Codec,
    CodecPipeline,
)
from zarr.core.config import config
from zarr.core.indexing import SelectorTuple, is_integer, ArrayIndexError
from zarr.registry import register_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Self

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.common import ChunkCoords

from ._internal import CodecPipelineImpl


# adapted from https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
def get_max_threads() -> int:
    return (os.cpu_count() or 1) + 4


# This is a copy of the function from zarr.core.indexing that fixes:
#   DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated
# TODO: Upstream this fix
def make_slice_selection(selection: Any) -> list[slice]:
    ls: list[slice] = []
    for dim_selection in selection:
        if is_integer(dim_selection):
            ls.append(slice(int(dim_selection), int(dim_selection) + 1, 1))
        elif isinstance(dim_selection, np.ndarray):
            if len(dim_selection) == 1:
                ls.append(slice(int(dim_selection.item()), int(dim_selection.item()) + 1, 1))
            else:
                raise ArrayIndexError
        else:
            ls.append(dim_selection)
    return ls


def selector_tuple_to_slice_selection(selector_tuple: SelectorTuple) -> list[slice]:
    return (
        [selector_tuple]
        if isinstance(selector_tuple, slice)
        else make_slice_selection(selector_tuple)
    )


def make_chunk_info_for_rust(
    batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]],
) -> list[tuple[str, ChunkCoords, str, Any, list[slice], list[slice]]]:
    return list(
        (
            str(byte_getter),
            chunk_spec.shape,
            str(chunk_spec.dtype),
            chunk_spec.fill_value.tobytes(),
            selector_tuple_to_slice_selection(out_selection),
            selector_tuple_to_slice_selection(chunk_selection),
        )
        for (byte_getter, chunk_spec, chunk_selection, out_selection) in batch_info
    )


@dataclass(frozen=True)
class ZarrsCodecPipeline(CodecPipeline):
    codecs: tuple[Codec, ...]
    impl: CodecPipelineImpl

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        raise NotImplementedError("evolve_from_array_spec")

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec]) -> Self:
        codec_metadata = [codec.to_dict() for codec in codecs]
        codec_metadata_json = json.dumps(codec_metadata)
        return cls(
            codecs=tuple(codecs),
            impl=CodecPipelineImpl(
                codec_metadata_json,
                config.get("codec_pipeline.validate_checksums", None),
                config.get("codec_pipeline.store_empty_chunks", None),
                config.get("codec_pipeline.concurrent_target", None),
            ),
        )

    @property
    def supports_partial_decode(self) -> bool:
        return False

    @property
    def supports_partial_encode(self) -> bool:
        return False

    def __iter__(self) -> Iterator[Codec]:
        yield from self.codecs

    def validate(
        self, *, shape: ChunkCoords, dtype: np.dtype[Any], chunk_grid: ChunkGrid
    ) -> None:
        raise NotImplementedError("validate")

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        raise NotImplementedError("compute_encoded_size")

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        raise NotImplementedError("decode")

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        raise NotImplementedError("encode")

    async def read(
        self,
        batch_info: Iterable[
            tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple]
        ],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),  # FIXME: unused
    ) -> None:
        chunk_concurrent_limit = (
            config.get("threading.max_workers") or get_max_threads()
        )
        out = out.as_ndarray_like()  # FIXME: Error if array is not in host memory
        if not out.dtype.isnative:
            raise RuntimeError("Non-native byte order not supported")

        chunks_desc = make_chunk_info_for_rust(batch_info)
        res = await asyncio.to_thread(
            self.impl.retrieve_chunks, chunks_desc, out, chunk_concurrent_limit
        )
        if drop_axes != ():
            res = res.squeeze(axis=drop_axes)
        return res

    async def write(
        self,
        batch_info: Iterable[
            tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]
        ],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        # FIXME: use drop_axes
        chunk_concurrent_limit = (
            config.get("threading.max_workers") or get_max_threads()
        )
        value = value.as_ndarray_like()  # FIXME: Error if array is not in host memory
        if not value.dtype.isnative:
            value = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("="))
        elif not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
        chunks_desc = make_chunk_info_for_rust(batch_info)
        return await asyncio.to_thread(
            self.impl.store_chunks, chunks_desc, value, chunk_concurrent_limit
        )


register_pipeline(ZarrsCodecPipeline)

__all__ = ["ZarrsCodecPipeline"]
