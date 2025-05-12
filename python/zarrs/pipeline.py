from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from zarr.abc.codec import Codec, CodecPipeline
from zarr.core import BatchedCodecPipeline
from zarr.core.config import config

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator
    from typing import Any, Self

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDArrayLike, NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.common import ChunkCoords
    from zarr.core.indexing import SelectorTuple

from ._internal import CodecPipelineImpl, codec_metadata_v2_to_v3
from .utils import (
    CollapsedDimensionError,
    DiscontiguousArrayError,
    FillValueNoneError,
    make_chunk_info_for_rust_with_indices,
)


class UnsupportedDataTypeError(Exception):
    pass


class UnsupportedMetadataError(Exception):
    pass


def get_codec_pipeline_impl(codec_metadata_json: str) -> CodecPipelineImpl | None:
    try:
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
    except TypeError as e:
        if re.match(r"codec (delta|zlib) is not supported", str(e)):
            return None
        else:
            raise e


def codecs_to_dict(codecs: Iterable[Codec]) -> Generator[dict[str, Any], None, None]:
    for codec in codecs:
        if codec.__class__.__name__ == "V2Codec":
            codec_dict = codec.to_dict()
            if codec_dict.get("filters", None) is not None:
                filters = [
                    json.dumps(filter.get_config())
                    for filter in codec_dict.get("filters")
                ]
            else:
                filters = None
            if codec_dict.get("compressor", None) is not None:
                compressor_json = codec_dict.get("compressor").get_config()
                compressor = json.dumps(compressor_json)
            else:
                compressor = None
            codecs_v3 = codec_metadata_v2_to_v3(filters, compressor)
            for codec in codecs_v3:
                yield json.loads(codec)
        else:
            yield codec.to_dict()


class ZarrsCodecPipelineState(TypedDict):
    codec_metadata_json: str
    codecs: tuple[Codec, ...]


@dataclass
class ZarrsCodecPipeline(CodecPipeline):
    codecs: tuple[Codec, ...]
    impl: CodecPipelineImpl | None
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
            tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]
        ],
        out: NDBuffer,  # type: ignore
        drop_axes: tuple[int, ...] = (),  # FIXME: unused
    ) -> None:
        # FIXME: Error if array is not in host memory
        if not out.dtype.isnative:
            raise RuntimeError("Non-native byte order not supported")
        try:
            if self.impl is None:
                raise UnsupportedMetadataError()
            self._raise_error_on_unsupported_batch_dtype(batch_info)
            chunks_desc = make_chunk_info_for_rust_with_indices(
                batch_info, drop_axes, out.shape
            )
        except (
            UnsupportedMetadataError,
            DiscontiguousArrayError,
            CollapsedDimensionError,
            UnsupportedDataTypeError,
            FillValueNoneError,
        ):
            await self.python_impl.read(batch_info, out, drop_axes)
            return None
        else:
            out: NDArrayLike = out.as_ndarray_like()
            await asyncio.to_thread(
                self.impl.retrieve_chunks_and_apply_index,
                chunks_desc,
                out,
            )
            return None

    async def write(
        self,
        batch_info: Iterable[
            tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]
        ],
        value: NDBuffer,  # type: ignore
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        try:
            if self.impl is None:
                raise UnsupportedMetadataError()
            self._raise_error_on_unsupported_batch_dtype(batch_info)
            chunks_desc = make_chunk_info_for_rust_with_indices(
                batch_info, drop_axes, value.shape
            )
        except (
            UnsupportedMetadataError,
            DiscontiguousArrayError,
            CollapsedDimensionError,
            UnsupportedDataTypeError,
            FillValueNoneError,
        ):
            await self.python_impl.write(batch_info, value, drop_axes)
            return None
        else:
            # FIXME: Error if array is not in host memory
            value_np: NDArrayLike | np.ndarray = value.as_ndarray_like()
            if not value_np.dtype.isnative:
                value_np = np.ascontiguousarray(
                    value_np, dtype=value_np.dtype.newbyteorder("=")
                )
            elif not value_np.flags.c_contiguous:
                value_np = np.ascontiguousarray(value_np)
            await asyncio.to_thread(
                self.impl.store_chunks_with_indices, chunks_desc, value_np
            )
            return None

    def _raise_error_on_unsupported_batch_dtype(
        self,
        batch_info: Iterable[
            tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]
        ],
    ):
        # https://github.com/LDeakin/zarrs/blob/0532fe983b7b42b59dbf84e50a2fe5e6f7bad4ce/zarrs_metadata/src/v2_to_v3.rs#L289-L293 for VSUMm
        # Further, our pipeline does not support variable-length objects due to limitations on decode_into, so object/np.dtypes.StringDType is also out
        if any(
            info.dtype.kind in {"V", "S", "U", "M", "m", "O", "T"}
            for (_, info, _, _, _) in batch_info
        ):
            raise UnsupportedDataTypeError()
