from __future__ import annotations

import operator
import os
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

from zarr.core.array_spec import ArraySpec

from zarrs._internal import ChunkItem

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.indexing import SelectorTuple
    from zarr.dtype import ZDType


# adapted from https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
def get_max_threads() -> int:
    return (os.cpu_count() or 1) + 4


class DiscontiguousArrayError(Exception):
    pass


class CollapsedDimensionError(Exception):
    pass


class FillValueNoneError(Exception):
    pass


class UnsupportedIndexType(Exception):
    pass


def selector_tuple_to_slice_selection(selector_tuple: SelectorTuple) -> list[slice]:
    if isinstance(selector_tuple, slice):
        return [selector_tuple]
    if all(isinstance(s, slice) for s in selector_tuple):
        return list(selector_tuple)
    raise UnsupportedIndexType(
        f"Invalid index type detected: {type(selector_tuple[0])}"
    )


def prod_op(x: Iterable[int]) -> int:
    return reduce(operator.mul, x, 1)


def get_implicit_fill_value(dtype: ZDType, fill_value: Any) -> Any:
    if fill_value is None:
        fill_value = dtype.default_scalar()
    return fill_value


@dataclass(frozen=True)
class RustChunkInfo:
    chunk_info_with_indices: list[ChunkItem]
    write_empty_chunks: bool


def make_chunk_info_for_rust_with_indices(
    batch_info: Iterable[
        tuple[ByteGetter | ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]
    ],
    drop_axes: tuple[int, ...],
    shape: tuple[int, ...],
) -> RustChunkInfo:
    shape = shape if shape else (1,)  # constant array
    chunk_info_with_indices: list[ChunkItem] = []
    write_empty_chunks: bool = True
    for (
        byte_getter,
        chunk_spec,
        chunk_selection,
        out_selection,
        _,
    ) in batch_info:
        write_empty_chunks = chunk_spec.config.write_empty_chunks
        if chunk_spec.fill_value is None:
            chunk_spec = ArraySpec(
                chunk_spec.shape,
                chunk_spec.dtype,
                get_implicit_fill_value(chunk_spec.dtype, chunk_spec.fill_value),
                chunk_spec.config,
                chunk_spec.prototype,
            )
        out_selection_as_slices = selector_tuple_to_slice_selection(out_selection)
        chunk_selection_as_slices = selector_tuple_to_slice_selection(chunk_selection)
        chunk_info_with_indices.append(
            ChunkItem(
                key=byte_getter.path,
                chunk_subset=chunk_selection_as_slices,
                chunk_shape=chunk_spec.shape,
                subset=out_selection_as_slices,
                shape=shape,
            )
        )
    return RustChunkInfo(chunk_info_with_indices, write_empty_chunks)
