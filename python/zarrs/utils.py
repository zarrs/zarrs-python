from __future__ import annotations

import operator
import os
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.core.indexing import is_integer

from zarrs._internal import ChunkItem

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import EllipsisType

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.indexing import SelectorTuple
    from zarr.dtype import ZDType


# adapted from https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
def get_max_threads() -> int:
    return (os.cpu_count() or 1) + 4


class DiscontiguousArrayError(Exception):
    pass


class UnsupportedVIndexingError(Exception):
    pass


class FillValueNoneError(Exception):
    pass


# This is a (mostly) copy of the function from zarr.core.indexing that fixes:
#   DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated
# TODO: Upstream this fix
def make_slice_selection(selection: tuple[np.ndarray | float]) -> list[slice]:
    ls: list[slice] = []
    for dim_selection in selection:
        if is_integer(dim_selection):
            ls.append(slice(int(dim_selection), int(dim_selection) + 1, 1))
        elif isinstance(dim_selection, np.ndarray):
            dim_selection = dim_selection.ravel()
            if len(dim_selection) == 1:
                ls.append(
                    slice(int(dim_selection.item()), int(dim_selection.item()) + 1, 1)
                )
            else:
                diff = np.diff(dim_selection)
                if (diff != 1).any() and (diff != 0).any():
                    raise DiscontiguousArrayError(diff)
                ls.append(slice(dim_selection[0], dim_selection[-1] + 1, 1))
        else:
            ls.append(dim_selection)
    return ls


def selector_tuple_to_slice_selection(selector_tuple: SelectorTuple) -> list[slice]:
    if isinstance(selector_tuple, slice):
        return [selector_tuple]
    if all(isinstance(s, slice) for s in selector_tuple):
        return list(selector_tuple)
    return make_slice_selection(selector_tuple)


def resulting_shape_from_index(
    array_shape: tuple[int, ...],
    index_tuple: tuple[int | slice | EllipsisType | np.ndarray],
    drop_axes: tuple[int, ...],
    *,
    pad: bool,
) -> tuple[int, ...]:
    result_shape = []
    advanced_index_shapes = [
        idx.shape for idx in index_tuple if isinstance(idx, np.ndarray)
    ]
    basic_shape_index = 0

    # Broadcast all advanced indices, if any
    if advanced_index_shapes:
        result_shape += np.broadcast_shapes(*advanced_index_shapes)
        # Consume dimensions from array_shape
        basic_shape_index += len(advanced_index_shapes)

    # Process each remaining index in index_tuple
    for idx in index_tuple:
        if isinstance(idx, int):
            # Integer index reduces dimension, so skip this dimension in array_shape
            basic_shape_index += 1
        elif isinstance(idx, slice):
            if idx.step is not None and idx.step > 1:
                raise DiscontiguousArrayError(
                    "Step size greater than 1 is not supported"
                )
            # Slice keeps dimension, adjust size accordingly
            start, stop, _ = idx.indices(array_shape[basic_shape_index])
            result_shape.append(stop - start)
            basic_shape_index += 1
        elif idx is Ellipsis:
            # Calculate number of dimensions that Ellipsis should fill
            num_to_fill = len(array_shape) - len(index_tuple) + 1
            result_shape += array_shape[
                basic_shape_index : basic_shape_index + num_to_fill
            ]
            basic_shape_index += num_to_fill
        elif not isinstance(idx, np.ndarray):
            raise ValueError(f"Invalid index type: {type(idx)}")

    # Step 4: Append remaining dimensions from array_shape if fewer indices were used
    if basic_shape_index < len(array_shape) and pad:
        result_shape += array_shape[basic_shape_index:]

    return tuple(size for idx, size in enumerate(result_shape) if idx not in drop_axes)


def prod_op(x: Iterable[int]) -> int:
    return reduce(operator.mul, x, 1)


def get_shape_for_selector(
    selector_tuple: SelectorTuple,
    shape: tuple[int, ...],
    *,
    pad: bool,
    drop_axes: tuple[int, ...] = (),
) -> tuple[int, ...]:
    if isinstance(selector_tuple, slice | np.ndarray):
        return resulting_shape_from_index(
            shape,
            (selector_tuple,),
            drop_axes,
            pad=pad,
        )
    return resulting_shape_from_index(shape, selector_tuple, drop_axes, pad=pad)


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
    is_constant = shape == ()
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
        out_selection_as_slices = selector_tuple_to_slice_selection(out_selection)
        chunk_selection_as_slices = selector_tuple_to_slice_selection(chunk_selection)
        shape_chunk_selection_slices = get_shape_for_selector(
            tuple(chunk_selection_as_slices),
            chunk_spec.shape,
            pad=True,
            drop_axes=drop_axes,
        )
        shape_chunk_selection = get_shape_for_selector(
            chunk_selection, chunk_spec.shape, pad=True, drop_axes=drop_axes
        )
        if (chunk_size := prod_op(shape_chunk_selection)) != prod_op(
            shape_chunk_selection_slices
        ):
            raise UnsupportedVIndexingError(
                f"{shape_chunk_selection} != {shape_chunk_selection_slices}"
            )
        if not is_constant and chunk_size > prod_op(shape):
            raise IndexError(
                f"the size of the chunk subset {chunk_size} and input/output subset {prod_op(shape)} are incompatible"
            )
        io_array_shape = list(shape)
        out_selection_expanded = out_selection_as_slices
        # We need to have io_array_shape and out_selection_expanded with dimensionalities matching that of the underlying array.
        # `drop_axes`` is only triggered via fancy outer-indexing, and everything else silently drops.
        # So if we detect that a dimension has been dropped silently (due to a singleton axis, like `z[1, ...]`) after converting to slices, we update these two values.
        if (
            not drop_axes
            and not is_constant
            and len(shape_chunk_selection) != len(shape_chunk_selection_slices)
        ):
            shape_ctr = 0
            for idx_shape, shape_chunk in enumerate(shape_chunk_selection_slices):
                # Detect if this dimension has been dropped on the io_array i.e., shape_chunk_selection has been exhausted so there is an extra 1-sized dimension at the end or has a mismatch with the "full" chunk shape `shape_chunk_selection_slices`.
                if shape_chunk == 1 and (
                    shape_ctr >= len(shape_chunk_selection)
                    or shape_chunk != shape_chunk_selection[shape_ctr]
                ):
                    drop_axes += (idx_shape,)
        if drop_axes:
            for axis in drop_axes:
                io_array_shape.insert(axis, 1)
                out_selection_expanded.insert(axis, slice(0, 1))
        chunk_info_with_indices.append(
            ChunkItem(
                key=byte_getter.path,
                chunk_subset=chunk_selection_as_slices,
                chunk_shape=chunk_spec.shape,
                subset=out_selection_expanded,
                shape=io_array_shape,
            )
        )
    return RustChunkInfo(chunk_info_with_indices, write_empty_chunks)
