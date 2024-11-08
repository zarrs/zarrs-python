from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.core.indexing import ArrayIndexError, SelectorTuple, is_integer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zarr.abc.store import ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import ChunkCoords


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
                ls.append(
                    slice(int(dim_selection.item()), int(dim_selection.item()) + 1, 1)
                )
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
