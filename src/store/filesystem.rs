use std::sync::Arc;

use pyo3::{exceptions::PyRuntimeError, pyclass, PyErr};
use pyo3_stub_gen::derive::gen_stub_pyclass;
use zarrs::{filesystem::FilesystemStore, storage::ReadableWritableListableStorageTraits};

use crate::utils::PyErrExt;

use super::StoreConfig;

#[gen_stub_pyclass]
#[pyclass(extends=StoreConfig)]
pub struct FilesystemStoreConfig {
    root: String,
}

impl FilesystemStoreConfig {
    pub fn new(root: String) -> Self {
        Self { root }
    }
}

impl TryInto<Arc<dyn ReadableWritableListableStorageTraits>> for &FilesystemStoreConfig {
    type Error = PyErr;

    fn try_into(self) -> Result<Arc<dyn ReadableWritableListableStorageTraits>, Self::Error> {
        let store: Arc<dyn ReadableWritableListableStorageTraits> =
            Arc::new(FilesystemStore::new(self.root.clone()).map_py_err::<PyRuntimeError>()?);
        Ok(store)
    }
}
