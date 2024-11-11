from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.core.indexing import SelectorTuple, is_integer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import ChunkCoords


# adapted from https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
def get_max_threads() -> int:
    return (os.cpu_count() or 1) + 4


class DiscontiguousArrayError(BaseException):
    pass


# This is a copy of the function from zarr.core.indexing that fixes:
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


def convert_chunk_to_primitive(
    byte_getter: ByteGetter | ByteSetter, chunk_spec: ArraySpec
) -> tuple[str, ChunkCoords, str, Any]:
    return (
        str(byte_getter),
        chunk_spec.shape,
        str(chunk_spec.dtype),
        chunk_spec.fill_value.tobytes(),
    )


def make_chunk_info_for_rust_with_indices(
    batch_info: Iterable[
        tuple[ByteGetter | ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]
    ],
) -> list[tuple[tuple[str, ChunkCoords, str, Any], list[slice], list[slice]]]:
    return list(
        (
            convert_chunk_to_primitive(byte_getter, chunk_spec),
            selector_tuple_to_slice_selection(out_selection),
            selector_tuple_to_slice_selection(chunk_selection),
        )
        for (byte_getter, chunk_spec, chunk_selection, out_selection) in batch_info
    )


def make_chunk_info_for_rust(
    batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple]],
) -> list[tuple[str, ChunkCoords, str, Any]]:
    return list(
        convert_chunk_to_primitive(byte_getter, chunk_spec)
        for (byte_getter, chunk_spec, chunk_selection, out_selection) in batch_info
    )
