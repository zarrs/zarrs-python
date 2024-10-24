from __future__ import annotations

from dataclasses import dataclass
from itertools import islice, pairwise
from typing import TYPE_CHECKING, Any, TypeVar
from warnings import warn
import numpy as np
import json

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    BytesBytesCodec,
    Codec,
    CodecPipeline,
)
from zarr.core.common import ChunkCoords, concurrent_map
from zarr.core.config import config
from zarr.core.indexing import SelectorTuple, is_scalar, is_total_slice
from zarr.registry import register_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Self

    import numpy as np

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
    from zarr.core.chunk_grids import ChunkGrid

from ._internal import CodecPipelineImpl


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
            impl=CodecPipelineImpl(codec_metadata_json),
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
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        # FIXME: use drop_axes
        # TODO: Instead of iterating here: add read_chunk_subsets to CodecPipelineImpl
        for byte_getter, chunk_spec, chunk_selection, out_selection in batch_info:
            chunk_path = str(byte_getter)

            if all(is_total_slice(sel, chunk_spec.shape) for sel in chunk_selection):
                np_array_chunk = self.impl.retrieve_chunk(chunk_path, chunk_spec.shape, str(chunk_spec.dtype), chunk_spec.fill_value.tobytes())
                out[out_selection] = np_array_chunk.view(chunk_spec.dtype).reshape(chunk_spec.shape)
            else:
                # FIXME: This does not do partial decoding
                np_array_chunk = self.impl.retrieve_chunk(chunk_path, chunk_spec.shape, str(chunk_spec.dtype), chunk_spec.fill_value.tobytes())
                out[out_selection] = np_array_chunk.view(chunk_spec.dtype).reshape(chunk_spec.shape)[chunk_selection]

                # np_array_chunk = self.impl.retrieve_chunk_subset(chunk_path, chunk_spec.shape, str(chunk_spec.dtype), chunk_selection, chunk_spec.fill_value.tobytes())
                # out[out_selection] = np_array_chunk.view(chunk_spec.dtype).reshape(chunk_spec.shape)

    async def write(
        self,
        batch_info: Iterable[
            tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]
        ],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        # FIXME: use drop_axes
        value = value.as_ndarray_like() # FIXME: Error if array is not in host memory

        # TODO: Instead of iterating here: add store_chunk_subsets to CodecPipelineImpl
        for byte_setter, chunk_spec, chunk_selection, out_selection in batch_info:
            chunk_path = str(byte_setter)

            # FIXME: Could pass bytes(value.data) directly to store_chunk_subset with out_selection to avoid copying, but all the indexing has to be handled on the rust side
            if all(is_total_slice(sel, value.shape) for sel in chunk_selection):
                if value.ndim == 0:
                    chunk_bytes = np.broadcast_to(value, chunk_spec.shape).tobytes() # copies
                elif value.flags.c_contiguous:
                    chunk_bytes = bytes(value.data) # 0-copy
                else:
                    chunk_bytes = value.tobytes() # copies
                self.impl.store_chunk(chunk_path, chunk_spec.shape, str(chunk_spec.dtype), chunk_spec.fill_value.tobytes(), chunk_bytes)
            elif all(is_total_slice(sel, chunk_spec.shape) for sel in chunk_selection):
                chunk_bytes = value[out_selection].tobytes() # copies
                self.impl.store_chunk(chunk_path, chunk_spec.shape, str(chunk_spec.dtype), chunk_spec.fill_value.tobytes(), chunk_bytes)
            else:
                # FIXME: Probably better to do this on the rust side, but then have to handle indexing
                chunk = self.impl.retrieve_chunk(chunk_path, chunk_spec.shape, str(chunk_spec.dtype), chunk_spec.fill_value.tobytes()).view(chunk_spec.dtype).reshape(chunk_spec.shape)
                chunk[chunk_selection] = value[out_selection]
                self.impl.store_chunk(chunk_path, chunk_spec.shape, str(chunk_spec.dtype), chunk_spec.fill_value.tobytes(), chunk.tobytes())

                # value_selection = value[out_selection]
                # if drop_axes != ():
                #     value_selection = np.squeeze(value_selection, axis=drop_axes)
                # chunk_bytes = value_selection.tobytes() # copies
                # self.impl.store_chunk_subset(chunk_path, chunk_spec.shape, str(chunk_spec.dtype), chunk_spec.fill_value.tobytes(), chunk_selection, chunk_bytes)

register_pipeline(ZarrsCodecPipeline)

__all__ = ["ZarrsCodecPipeline"]