#!/usr/bin/env python3

import zarrs_python  # noqa: F401
import zarr
from zarr.storage import LocalStore, MemoryStore
import tempfile

zarr.config.set(codec_pipeline={"path": "zarrs_python.ZarrsCodecPipeline",})

tmp = tempfile.TemporaryDirectory()
arr = zarr.zeros((100,), store=LocalStore(root=tmp.name, mode='w'), chunks=(10,), dtype='i4', codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()])

# arr = zarr.zeros((100,), store=MemoryStore(mode='w'), chunks=(10,), dtype='u1', codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()])

arr[:] = 42
print(arr[:])
