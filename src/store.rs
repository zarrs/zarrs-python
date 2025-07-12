use std::{collections::HashMap, sync::Arc};

use opendal::Builder;
use pyo3::{
    exceptions::{PyNotImplementedError, PyValueError},
    types::{PyAnyMethods, PyStringMethods, PyTypeMethods},
    Bound, FromPyObject, PyAny, PyErr, PyResult,
};
use pyo3_stub_gen::derive::gen_stub_pyclass_enum;
use zarrs::storage::{
    storage_adapter::async_to_sync::AsyncToSyncStorageAdapter, ReadableWritableListableStorage,
};

use crate::{runtime::tokio_block_on, utils::PyErrExt};

mod filesystem;
mod http;

pub use self::filesystem::FilesystemStoreConfig;
pub use self::http::HttpStoreConfig;

#[derive(Debug, Clone)]
#[gen_stub_pyclass_enum]
pub enum StoreConfig {
    Filesystem(FilesystemStoreConfig),
    Http(HttpStoreConfig),
    // TODO: Add support for more stores
}

impl<'py> FromPyObject<'py> for StoreConfig {
    fn extract_bound(store: &Bound<'py, PyAny>) -> PyResult<Self> {
        let name = store.get_type().name()?;
        let name = name.to_str()?;
        match name {
            "LocalStore" => {
                let root: String = store.getattr("root")?.call_method0("__str__")?.extract()?;
                Ok(StoreConfig::Filesystem(FilesystemStoreConfig::new(root)))
            }
            "FsspecStore" => {
                let fs = store.getattr("fs")?;
                let fs_name = fs.get_type().name()?;
                let fs_name = fs_name.to_str()?;
                let path: String = store.getattr("path")?.extract()?;
                let storage_options: HashMap<String, Bound<'py, PyAny>> =
                    fs.getattr("storage_options")?.extract()?;
                match fs_name {
                    "HTTPFileSystem" => Ok(StoreConfig::Http(HttpStoreConfig::new(
                        &path,
                        &storage_options,
                    )?)),
                    _ => Err(PyErr::new::<PyNotImplementedError, _>(format!(
                        "zarrs-python does not support {fs_name} (FsspecStore) stores"
                    ))),
                }
            }
            _ => Err(PyErr::new::<PyNotImplementedError, _>(format!(
                "zarrs-python does not support {name} stores"
            ))),
        }
    }
}

impl TryFrom<&StoreConfig> for ReadableWritableListableStorage {
    type Error = PyErr;

    fn try_from(value: &StoreConfig) -> Result<Self, Self::Error> {
        match value {
            StoreConfig::Filesystem(config) => config.try_into(),
            StoreConfig::Http(config) => config.try_into(),
        }
    }
}

fn opendal_builder_to_sync_store<B: Builder>(
    builder: B,
) -> PyResult<ReadableWritableListableStorage> {
    let operator = opendal::Operator::new(builder)
        .map_py_err::<PyValueError>()?
        .finish();
    let store = Arc::new(zarrs_opendal::AsyncOpendalStore::new(operator));
    let store = Arc::new(AsyncToSyncStorageAdapter::new(store, tokio_block_on()));
    Ok(store)
}
