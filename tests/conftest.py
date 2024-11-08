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

import zarrs_python  # noqa: F401

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
    config.set({"codec_pipeline.path": "zarrs_python.ZarrsCodecPipeline"})


async def parse_store(
    store: Literal["local", "memory", "remote", "zip"], path: str
) -> LocalStore | MemoryStore | RemoteStore | ZipStore:
    if store == "local":
        return await LocalStore.open(path, mode="w")
    if store == "memory":
        return await MemoryStore.open(mode="w")
    if store == "remote":
        return await RemoteStore.open(url=path, mode="w")
    if store == "zip":
        return await ZipStore.open(path + "/zarr.zip", mode="w")
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
    "test_roundtrip[oindex-contiguous_in_chunk_array-ellipsis]",
    "test_roundtrip[oindex-discontinuous_in_chunk_array-ellipsis]",
    "test_roundtrip[vindex-contiguous_in_chunk_array-ellipsis]",
    "test_roundtrip[vindex-discontinuous_in_chunk_array-ellipsis]",
    "test_roundtrip[oindex-across_chunks_indices_array-ellipsis]",
    "test_roundtrip[vindex-ellipsis-across_chunks_indices_array]",
    "test_roundtrip[vindex-across_chunks_indices_array-ellipsis]",
    "test_roundtrip[vindex-ellipsis-contiguous_in_chunk_array]",
    "test_roundtrip[vindex-ellipsis-discontinuous_in_chunk_array]",
]


def pytest_collection_modifyitems(
    config: pytest.Config, items: Iterable[pytest.Item]
) -> None:
    xfail_marker = pytest.mark.xfail(
        reason="These tests fail with the zarr-python default codec pipeline."
    )
    for item in items:
        if item.name in zarr_python_default_codec_pipeline_failures:
            item.add_marker(xfail_marker)
