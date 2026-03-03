import platform
import subprocess
import tempfile
import time

import numpy as np
import zarr


def clear_cache():
    if platform.system() == "Darwin":
        subprocess.call(["sync", "&&", "sudo", "purge"])
    elif platform.system() == "Linux":
        subprocess.call(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
    else:
        raise Exception("Unsupported platform")

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

# zarr.config.set({"codec_pipeline.decode_mode": "auto"})
# full read took:  3.3279900550842285
# partial shard read (4095) took:  1.211921215057373
# partial shard read (4096) took:  2.3402509689331055

zarr.config.set({"codec_pipeline.decode_mode": "partial"})
# full read took:  2.2892508506774902
# partial shard read (4095) took:  1.1934266090393066
# partial shard read (4096) took:  1.1788337230682373

z = zarr.create_array(
    "examples/issue_152.zarr",
    shape=(8192, 4, 128, 128),
    shards=(4096, 4, 128, 128),
    chunks=(1, 1, 128, 128),
    dtype=np.float64,
    overwrite=True,
)
z[...] = np.random.randn(8192, 4, 128, 128)

clear_cache()
t = time.time()
z[...]
print("full read took: ", time.time() - t)

clear_cache()
t = time.time()
z[:4095, ...]
print("partial shard read (4095) took: ", time.time() - t)


clear_cache()
t = time.time()
z[:4096, ...]
print("partial shard read (4096) took: ", time.time() - t)
