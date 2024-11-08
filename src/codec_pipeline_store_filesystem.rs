use std::sync::Arc;

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    PyErr, PyResult,
};
use zarrs::{filesystem::FilesystemStore, storage::ReadableWritableListableStorageTraits};

use crate::{utils::PyErrExt, CodecPipelineStore};

pub struct CodecPipelineStoreFilesystem {
    store: Arc<FilesystemStore>,
    cwd: String,
}

impl CodecPipelineStoreFilesystem {
    pub fn new() -> PyResult<Self> {
        let store = Arc::new(FilesystemStore::new("/").map_py_err::<PyRuntimeError>()?);
        let cwd = std::env::current_dir()?
            .to_string_lossy()
            .replace('\\', "/"); // TODO: Check zarr-python path handling on windows

        // Remove the leading / from the cwd if preset, so cwd is a valid Zarr store path
        let cwd = cwd.strip_prefix("/").unwrap_or(&cwd).to_string();
        Ok(Self { store, cwd })
    }
}

impl CodecPipelineStore for CodecPipelineStoreFilesystem {
    fn store(&self) -> Arc<dyn ReadableWritableListableStorageTraits> {
        self.store.clone()
    }

    fn chunk_path(&self, store_path: &str) -> PyResult<String> {
        if let Some(chunk_path) = store_path.strip_prefix("file://") {
            if let Some(chunk_path) = chunk_path.strip_prefix("/") {
                Ok(chunk_path.to_string())
            } else {
                Ok(format!("{}/{}", self.cwd, chunk_path))
            }
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "a filesystem store was initialised, but received a store path without a file:// prefix: {store_path}"
            )))
        }
    }
}
