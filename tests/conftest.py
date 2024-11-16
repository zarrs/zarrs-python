from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pytest
from zarr import config
from zarr.core.common import ChunkCoords
from zarr.storage import LocalStore, MemoryStore, ZipStore
from zarr.storage.remote import RemoteStore

from zarrs.utils import (  # noqa: F401
    CollapsedDimensionError,
    DiscontiguousArrayError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Literal

    from zarr.abc.store import Store
    from zarr.core.common import ChunkCoords, MemoryOrder


@dataclass
class ArrayRequest:
    shape: ChunkCoords
    dtype: str
    order: MemoryOrder


@pytest.fixture(autouse=True)
def _setup_codec_pipeline():
    config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


async def parse_store(
    store: Literal["local", "memory", "remote", "zip"], path: str
) -> LocalStore | MemoryStore | RemoteStore | ZipStore:
    if store == "local":
        return await LocalStore.open(path)
    if store == "memory":
        return await MemoryStore.open()
    if store == "remote":
        return await RemoteStore.open(url=path)
    if store == "zip":
        return await ZipStore.open(path + "/zarr.zip")
    raise AssertionError


@pytest.fixture(params=["local"])
async def store(request: pytest.FixtureRequest, tmpdir) -> Store:
    param = request.param
    return await parse_store(param, str(tmpdir))


@pytest.fixture
def array_fixture(request: pytest.FixtureRequest) -> npt.NDArray[Any]:
    array_request: ArrayRequest = request.param
    return (
        np.arange(np.prod(array_request.shape))
        .reshape(array_request.shape, order=array_request.order)
        .astype(array_request.dtype)
    )


# tests that also fail with zarr-python's default codec pipeline
zarr_python_default_codec_pipeline_failures = [
    # ellipsis weirdness, need to report
    "test_roundtrip[oindex-2d-contiguous_in_chunk_array-ellipsis]",
    "test_roundtrip[oindex-2d-discontinuous_in_chunk_array-ellipsis]",
    "test_roundtrip[vindex-2d-contiguous_in_chunk_array-ellipsis]",
    "test_roundtrip[vindex-2d-discontinuous_in_chunk_array-ellipsis]",
    "test_roundtrip[oindex-2d-across_chunks_indices_array-ellipsis]",
    "test_roundtrip[vindex-2d-ellipsis-across_chunks_indices_array]",
    "test_roundtrip[vindex-2d-across_chunks_indices_array-ellipsis]",
    "test_roundtrip[vindex-2d-ellipsis-contiguous_in_chunk_array]",
    "test_roundtrip[vindex-2d-ellipsis-discontinuous_in_chunk_array]",
    "test_roundtrip_read_only_zarrs[oindex-2d-contiguous_in_chunk_array-ellipsis]",
    "test_roundtrip_read_only_zarrs[oindex-2d-discontinuous_in_chunk_array-ellipsis]",
    "test_roundtrip_read_only_zarrs[vindex-2d-contiguous_in_chunk_array-ellipsis]",
    "test_roundtrip_read_only_zarrs[vindex-2d-discontinuous_in_chunk_array-ellipsis]",
    "test_roundtrip_read_only_zarrs[oindex-2d-across_chunks_indices_array-ellipsis]",
    "test_roundtrip_read_only_zarrs[vindex-2d-ellipsis-across_chunks_indices_array]",
    "test_roundtrip_read_only_zarrs[vindex-2d-across_chunks_indices_array-ellipsis]",
    "test_roundtrip_read_only_zarrs[vindex-2d-ellipsis-contiguous_in_chunk_array]",
    "test_roundtrip_read_only_zarrs[vindex-2d-ellipsis-discontinuous_in_chunk_array]",
    # need to investigate this one - it seems to fail with the default pipeline
    # but it makes some sense that it succeeds with ours since we fall-back to numpy indexing
    # in the case of a collapsed dimension
    # "test_roundtrip_read_only_zarrs[vindex-2d-contiguous_in_chunk_array-contiguous_in_chunk_array]",
]

zarrs_python_no_discontinuous_writes = [
    "test_roundtrip[oindex-2d-discontinuous_in_chunk_array-slice_in_chunk]",
    "test_roundtrip[oindex-2d-discontinuous_in_chunk_array-slice_across_chunks]",
    "test_roundtrip[oindex-2d-discontinuous_in_chunk_array-full_slice]",
    "test_roundtrip[oindex-2d-discontinuous_in_chunk_array-int]",
    "test_roundtrip[oindex-2d-slice_in_chunk-discontinuous_in_chunk_array]",
    "test_roundtrip[oindex-2d-slice_across_chunks-discontinuous_in_chunk_array]",
    "test_roundtrip[oindex-2d-full_slice-discontinuous_in_chunk_array]",
    "test_roundtrip[oindex-2d-int-discontinuous_in_chunk_array]",
    "test_roundtrip[oindex-2d-ellipsis-discontinuous_in_chunk_array]",
    "test_roundtrip[vindex-2d-discontinuous_in_chunk_array-slice_in_chunk]",
    "test_roundtrip[vindex-2d-discontinuous_in_chunk_array-slice_across_chunks]",
    "test_roundtrip[vindex-2d-discontinuous_in_chunk_array-full_slice]",
    "test_roundtrip[vindex-2d-discontinuous_in_chunk_array-int]",
    "test_roundtrip[vindex-2d-slice_in_chunk-discontinuous_in_chunk_array]",
    "test_roundtrip[vindex-2d-slice_across_chunks-discontinuous_in_chunk_array]",
    "test_roundtrip[vindex-2d-full_slice-discontinuous_in_chunk_array]",
    "test_roundtrip[vindex-2d-int-discontinuous_in_chunk_array]",
    "test_roundtrip[oindex-2d-discontinuous_in_chunk_array-contiguous_in_chunk_array]",
    "test_roundtrip[oindex-2d-contiguous_in_chunk_array-discontinuous_in_chunk_array]",
    "test_roundtrip[oindex-2d-across_chunks_indices_array-discontinuous_in_chunk_array]",
    "test_roundtrip[oindex-2d-discontinuous_in_chunk_array-discontinuous_in_chunk_array]",
    "test_roundtrip[vindex-2d-contiguous_in_chunk_array-discontinuous_in_chunk_array]",
    "test_roundtrip[vindex-2d-discontinuous_in_chunk_array-discontinuous_in_chunk_array]",
    "test_roundtrip[oindex-2d-discontinuous_in_chunk_array-across_chunks_indices_array]",
    "test_roundtrip[vindex-2d-discontinuous_in_chunk_array-contiguous_in_chunk_array]",
    "test_roundtrip[oindex-1d-discontinuous_in_chunk_array]",
    "test_roundtrip[vindex-1d-discontinuous_in_chunk_array]",
]

# vindexing with two contiguous arrays would be converted to two slices but
# in numpy indexing actually requires dropping a dimension, which in turn boils
# down to integer indexing, which we can't do i.e., [np.array(1, 2), np.array(1, 2)] -> [slice(1, 3), slice(1, 3)]
# is not a correct conversion, and thus we don't support the write operation
zarrs_python_no_collapsed_dim = [
    "test_roundtrip[vindex-2d-contiguous_in_chunk_array-contiguous_in_chunk_array]"
]


def pytest_collection_modifyitems(
    config: pytest.Config, items: Iterable[pytest.Item]
) -> None:
    for item in items:
        if item.name in zarr_python_default_codec_pipeline_failures:
            xfail_marker = pytest.mark.xfail(
                reason="This test fails with the zarr-python default codec pipeline."
            )
            item.add_marker(xfail_marker)
        if item.name in zarrs_python_no_discontinuous_writes:
            xfail_marker = pytest.mark.xfail(
                raises=DiscontiguousArrayError,
                reason="zarrs discontinuous writes are not supported.",
            )
            item.add_marker(xfail_marker)
        if item.name in zarrs_python_no_collapsed_dim:
            xfail_marker = pytest.mark.xfail(
                raises=CollapsedDimensionError,
                reason="zarrs vindexing with multiple contiguous arrays is not supported.",
            )
            item.add_marker(xfail_marker)
