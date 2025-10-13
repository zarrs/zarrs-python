use std::sync::Arc;

use pyo3::{exceptions::PyRuntimeError, PyErr};
use zarrs::{
    filesystem::{FilesystemStore, FilesystemStoreOptions},
    storage::ReadableWritableListableStorage,
};

use crate::utils::PyErrExt;

#[derive(Debug, Clone)]
pub struct FilesystemStoreConfig {
    pub root: String,
    opts: FilesystemStoreOptions,
}

impl FilesystemStoreConfig {
    pub fn new(root: String) -> Self {
        Self {
            root,
            opts: FilesystemStoreOptions::default(),
        }
    }

    pub fn direct_io(&mut self, flag: bool) -> () {
        self.opts.direct_io(flag);
    }
}

impl TryInto<ReadableWritableListableStorage> for &FilesystemStoreConfig {
    type Error = PyErr;

    fn try_into(self) -> Result<ReadableWritableListableStorage, Self::Error> {
        let store = Arc::new(
            FilesystemStore::new_with_options(self.root.clone(), self.opts.clone())
                .map_py_err::<PyRuntimeError>()?,
        );
        Ok(store)
    }
}
