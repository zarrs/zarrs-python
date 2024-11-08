use std::sync::Arc;

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    PyErr, PyResult,
};
use zarrs::{filesystem::FilesystemStore, storage::ReadableWritableListableStorageTraits};

use crate::{utils::PyErrExt, CodecPipelineStore};

pub struct CodecPipelineStoreFilesystem {
    store: Arc<FilesystemStore>,
}

impl CodecPipelineStoreFilesystem {
    pub fn new() -> PyResult<Self> {
        let store = Arc::new(FilesystemStore::new("/").map_py_err::<PyRuntimeError>()?);
        Ok(Self { store })
    }
}

impl CodecPipelineStore for CodecPipelineStoreFilesystem {
    fn store(&self) -> Arc<dyn ReadableWritableListableStorageTraits> {
        self.store.clone()
    }

    fn chunk_path<'a>(&self, store_path: &'a str) -> PyResult<&'a str> {
        if let Some(chunk_path) = store_path.strip_prefix("file://") {
            Ok(chunk_path.strip_prefix("/").unwrap_or(chunk_path))
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "a filesystem store was initialised, but received a store path without a file:// prefix: {store_path}"
            )))
        }
    }
}
