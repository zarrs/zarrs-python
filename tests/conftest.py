from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import pytest

from zarr.core.common import ChunkCoords, MemoryOrder

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from zarr import config
from zarr.abc.store import Store
from zarr.storage import LocalStore, MemoryStore, ZipStore
from zarr.storage.remote import RemoteStore

import zarrs_python # noqa

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, Literal


    from zarr.core.common import ChunkCoords, MemoryOrder, ZarrFormat

@dataclass
class ArrayRequest:
    shape: ChunkCoords
    dtype: str
    order: MemoryOrder

@pytest.fixture(autouse=True)
def run_around_each_test():
    config.set({ "codec_pipeline.path": "zarrs_python.ZarrsCodecPipeline"})
    yield
    config.set({ "codec_pipeline.path": "zarrs_python.ZarrsCodecPipeline"})

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