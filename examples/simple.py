#!/usr/bin/env python3

import zarrs_python  # noqa: F401
import zarr
from zarr.storage import LocalStore, MemoryStore
import tempfile
import numpy as np

zarr.config.set(codec_pipeline={"path": "zarrs_python.ZarrsCodecPipeline",})

chunks = (2,) # FIXME: nd
shape = (4,)

tmp = tempfile.TemporaryDirectory()
arr = zarr.zeros(shape, store=LocalStore(root=tmp.name, mode='w'), chunks=chunks, dtype=np.uint8, codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()])

# arr = zarr.zeros(shape, store=MemoryStore(mode='w'), chunks=chunks, dtype=np.uint8, codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()])

arr[:] = 42
print(arr[:])
