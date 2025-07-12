use std::sync::Arc;

use pyo3::{exceptions::PyRuntimeError, pyclass, PyErr};
use pyo3_stub_gen::derive::gen_stub_pyclass;
use zarrs::{filesystem::FilesystemStore, storage::ReadableWritableListableStorage};

use crate::utils::PyErrExt;

#[derive(Debug, Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub struct FilesystemStoreConfig {
    #[pyo3(get, set)]
    pub root: String,
}

impl FilesystemStoreConfig {
    pub fn new(root: String) -> Self {
        Self { root }
    }
}

impl TryInto<ReadableWritableListableStorage> for &FilesystemStoreConfig {
    type Error = PyErr;

    fn try_into(self) -> Result<ReadableWritableListableStorage, Self::Error> {
        let store =
            Arc::new(FilesystemStore::new(self.root.clone()).map_py_err::<PyRuntimeError>()?);
        Ok(store)
    }
}
