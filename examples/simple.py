#!/usr/bin/env python3

import zarrs_python  # noqa: F401
import zarr
from zarr.storage import LocalStore, MemoryStore
import tempfile
import numpy as np

zarr.config.set(codec_pipeline={"path": "zarrs_python.ZarrsCodecPipeline",})

chunks = (2,2)
shape = (4,4)

tmp = tempfile.TemporaryDirectory()
arr = zarr.zeros(shape, store=LocalStore(root=tmp.name, mode='w'), chunks=chunks, dtype=np.int16, codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()])

# arr = zarr.zeros(shape, store=MemoryStore(mode='w'), chunks=chunks, dtype=np.uint8, codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()])

# Check fill value decoding
print(arr[:])
assert np.all(arr[:] == 0)

# Store a constant
arr[:] = 42
print(arr[:])
assert np.all(arr[:] == 42)

# Store a numpy array
arr[:] = np.arange(16).reshape(4,4)
print(arr[:])
assert np.all(arr[:] == np.arange(16).reshape(4,4))

# Decode a chunk
print(arr[0:2,0:2])
assert np.all(arr[0:2,0:2] == np.array([[0, 1], [4, 5]]))

# Partial decoding
print(arr[0:1,1:2])
assert np.all(arr[0:1,1:2] == np.array([[1]]))
print(arr[1:3,1:3])
assert np.all(arr[1:3,1:3] == np.array([[5, 6], [9, 10]]))

# Partial encoding
# TODO
arr[1:3,1:3] = np.array([[-1, -2], [-3, -4]])
print(arr[:])
assert np.all(arr[:] == np.array([[0, 1, 2, 3], [4, -1, -2, 7], [8, -3, -4, 11], [12, 13, 14, 15]]))
