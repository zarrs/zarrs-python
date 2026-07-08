import numpy as np
import pytest
from zarr import create_array
from zarr.abc.store import Store
from zarr.codecs import ZstdCodec
from zarr.storage import StorePath


@pytest.mark.parametrize("checksum", [True, False])
def test_zstd(*, store: Store, checksum: bool) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = create_array(
        StorePath(store, path="zstd"),
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        compressors=[ZstdCodec(level=0, checksum=checksum)],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])
