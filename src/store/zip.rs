use std::sync::Arc;

use pyo3::{PyErr, exceptions::PyIOError};
use zarrs::{
    filesystem::{FilesystemStore, FilesystemStoreOptions},
    storage::{ReadableListableStorage, StoreKey},
};
use zarrs_zip::ZipStorageAdapter;

use crate::utils::PyErrExt;

#[derive(Debug, Clone)]
pub struct ZipStoreConfig {
    pub path: String,
    opts: FilesystemStoreOptions,
}

impl ZipStoreConfig {
    pub fn new(path: String) -> Self {
        Self {
            path,
            opts: FilesystemStoreOptions::default(),
        }
    }

    pub fn direct_io(&mut self, flag: bool) -> () {
        self.opts.direct_io(flag);
    }
}

impl TryInto<ReadableListableStorage> for &ZipStoreConfig {
    type Error = PyErr;

    fn try_into(self) -> Result<ReadableListableStorage, Self::Error> {
        let path = std::path::Path::new(&self.path);
        // We get the directory containing the zip and the zip filename
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| PyIOError::new_err("Invalid ZIP path"))?;

        // Create the base filesystem store with options (including direct_io if enabled)
        // This improves performance when reading the ZIP file from disk
        let parent_str = parent.to_string_lossy().to_string();
        let base_store = Arc::new(
            FilesystemStore::new_with_options(parent_str, self.opts.clone())
                .map_py_err::<PyIOError>()?,
        );

        // Create StoreKey from filename
        let store_key = StoreKey::new(filename)
            .map_err(|e| PyIOError::new_err(format!("Invalid store key: {}", e)))?;

        // Wrap it in the Zip adapter
        // ZipStorageAdapter only supports read operations (ReadableListableStorage)
        // 
        // PERFORMANCE NOTE: zarrs_zip (v0.4.0) may be slower than Python's zipfile module
        // for ZIP stores due to:
        // 1. Less mature optimizations (Python's zipfile is highly optimized C code)
        // 2. Potential lack of central directory caching
        // 3. Additional indirection layers
        // For best performance with ZIP files, consider extracting and using a regular
        // filesystem store instead.
        //
        // Note: zarrs_zip has limitations - it doesn't support seekable compressed files
        let zip_store = Arc::new(
            ZipStorageAdapter::new(base_store, store_key)
                .map_err(|e| {
                    let error_msg = e.to_string();
                    // Provide a more helpful error message for common issues
                    if error_msg.contains("Seekable compressed files") {
                        PyIOError::new_err(format!(
                            "ZIP archive not supported: {}. \
                            The zarrs_zip library currently only supports ZIP files with \
                            uncompressed or non-seekable compressed entries. \
                            Consider using a different compression method or extracting the ZIP file.",
                            error_msg
                        ))
                    } else {
                        PyIOError::new_err(format!("ZIP archive error: {}", error_msg))
                    }
                })?,
        );

        Ok(zip_store)
    }
}
