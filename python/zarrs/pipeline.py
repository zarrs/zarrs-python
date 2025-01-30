from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, TypedDict

import numpy as np
from zarr.abc.codec import Codec, CodecPipeline
from zarr.codecs import BytesCodec
from zarr.core import BatchedCodecPipeline
from zarr.core.config import config

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any, Self

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDArrayLike, NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.common import ChunkCoords
    from zarr.core.indexing import SelectorTuple

from ._internal import CodecPipelineImpl
from .utils import (
    CollapsedDimensionError,
    DiscontiguousArrayError,
    make_chunk_info_for_rust,
    make_chunk_info_for_rust_with_indices,
)


def get_codec_pipeline_impl(codec_metadata_json: str) -> CodecPipelineImpl:
    return CodecPipelineImpl(
        codec_metadata_json,
        validate_checksums=config.get("codec_pipeline.validate_checksums", None),
        store_empty_chunks=config.get("array.write_empty_chunks", None),
        chunk_concurrent_minimum=config.get(
            "codec_pipeline.chunk_concurrent_minimum", None
        ),
        chunk_concurrent_maximum=config.get(
            "codec_pipeline.chunk_concurrent_maximum", None
        ),
        num_threads=config.get("threading.max_workers", None),
    )

def codecs_to_dict(codecs: Iterable[Codec]) -> Generator[dict[str, Any], None, None]:
    for codec in codecs:
        if codec.__class__.__name__ == "V2Codec":
            compressor = codec.to_dict()["compressor"].get_config()
            if compressor["id"] == "zstd":
                yield { "name": "zstd", "configuration": { "level": compressor["level"], "checksum": compressor["checksum"] } }
            # TODO: get the endianness added to V2Codec API
            yield BytesCodec().to_dict()
        else:
            yield codec.to_dict()

class ZarrsCodecPipelineState(TypedDict):
    codec_metadata_json: str
    codecs: tuple[Codec, ...]


@dataclass
class ZarrsCodecPipeline(CodecPipeline):
    codecs: tuple[Codec, ...]
    impl: CodecPipelineImpl
    codec_metadata_json: str
    python_impl: BatchedCodecPipeline

    def __getstate__(self) -> ZarrsCodecPipelineState:
        return {"codec_metadata_json": self.codec_metadata_json, "codecs": self.codecs}

    def __setstate__(self, state: ZarrsCodecPipelineState):
        self.codecs = state["codecs"]
        self.codec_metadata_json = state["codec_metadata_json"]
        self.impl = get_codec_pipeline_impl(self.codec_metadata_json)
        self.python_impl = BatchedCodecPipeline.from_codecs(self.codecs)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        raise NotImplementedError("evolve_from_array_spec")

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec]) -> Self:
        codec_metadata = list(codecs_to_dict(codecs))
        codec_metadata_json = json.dumps(codec_metadata)
        # TODO: upstream zarr-python has not settled on how to deal with configs yet
        # Should they be checked when an array is created, or when an operation is performed?
        # https://github.com/zarr-developers/zarr-python/issues/2409
        # https://github.com/zarr-developers/zarr-python/pull/2429#issuecomment-2566976567
        return cls(
            codec_metadata_json=codec_metadata_json,
            codecs=tuple(codecs),
            impl=get_codec_pipeline_impl(codec_metadata_json),
            python_impl=BatchedCodecPipeline.from_codecs(codecs),
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
        out: NDBuffer,  # type: ignore
        drop_axes: tuple[int, ...] = (),  # FIXME: unused
    ) -> None:
        out: NDArrayLike = out.as_ndarray_like()
        if not out.dtype.isnative:
            raise RuntimeError("Non-native byte order not supported")
        try:
            chunks_desc = make_chunk_info_for_rust_with_indices(
                batch_info, drop_axes, out.shape
            )
        except (DiscontiguousArrayError, CollapsedDimensionError):
            chunks_desc = make_chunk_info_for_rust(batch_info)
        else:
            await asyncio.to_thread(
                self.impl.retrieve_chunks_and_apply_index,
                chunks_desc,
                out,
            )
            return None
        chunks = await asyncio.to_thread(self.impl.retrieve_chunks, chunks_desc)
        for chunk, (_, spec, selection, out_selection) in zip(chunks, batch_info):
            chunk_reshaped = chunk.view(spec.dtype).reshape(spec.shape)
            chunk_selected = chunk_reshaped[selection]
            if drop_axes:
                chunk_selected = np.squeeze(chunk_selected, axis=drop_axes)
            out[out_selection] = chunk_selected

    async def write(
        self,
        batch_info: Iterable[
            tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]
        ],
        value: NDBuffer,  # type: ignore
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        # FIXME: Error if array is not in host memory
        value: NDArrayLike | np.ndarray = value.as_ndarray_like()
        if not value.dtype.isnative:
            value = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("="))
        elif not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
        chunks_desc = make_chunk_info_for_rust_with_indices(
            batch_info, drop_axes, value.shape
        )
        await asyncio.to_thread(self.impl.store_chunks_with_indices, chunks_desc, value)
        return None
