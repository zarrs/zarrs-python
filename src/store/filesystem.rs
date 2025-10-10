use std::sync::Arc;

use pyo3::{exceptions::PyRuntimeError, PyErr};
use zarrs::{filesystem::FilesystemStore, storage::ReadableWritableListableStorage};

use crate::map_py_err::PyErrStrExt as _;

#[derive(Debug, Clone)]
pub struct FilesystemStoreConfig {
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
        let store = Arc::new(
            FilesystemStore::new(self.root.clone()).map_py_err_from_str::<PyRuntimeError>()?,
        );
        Ok(store)
    }
}
