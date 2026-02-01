import tempfile
import warnings

import numpy as np
import zarr
from obstore.store import HTTPStore, LocalStore
from zarr.storage import ObjectStore

from .test_zarrs_http import ARR_REF, URL

# Suppress the expected warning about cross-library object store usage
warnings.filterwarnings(
    "ignore",
    message="Successfully reconstructed a store defined in another Python module",
)


def test_obstore_local_store():
    """Test zarrs-python with obstore LocalStore"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create obstore LocalStore and wrap with zarr's ObjectStore
        obstore_local = LocalStore(prefix=tmpdir)
        store = ObjectStore(obstore_local, read_only=False)

        # Create zarr array with obstore
        # Expect a warning about cross-library object store usage
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            arr = zarr.open_array(
                store=store,
                mode="w",
                shape=(100, 100),
                chunks=(10, 10),
                dtype="f4",
            )

            # Write data
            data = np.random.rand(100, 100).astype("f4")
            arr[:] = data

            # Read back and verify
            arr2 = zarr.open_array(store=store, mode="r")
            assert arr2.shape == (100, 100)
            assert np.allclose(arr2[:], data)


def test_obstore_http():
    """Test zarrs-python with obstore HTTPStore - similar to test_zarrs_http"""
    # Create HTTPStore from the test URL
    http_store = HTTPStore.from_url(URL)
    store = ObjectStore(http_store, read_only=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arr = zarr.open_array(store=store, mode="r")

        # Verify shape and data match the reference
        assert arr.shape == (8, 8)
        assert np.allclose(arr[:], ARR_REF, equal_nan=True)
