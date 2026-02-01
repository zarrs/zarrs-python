use std::{collections::HashMap, sync::Arc};

use opendal::Builder;
use pyo3::{
    Borrowed, Bound, FromPyObject, PyAny, PyErr, PyResult,
    exceptions::{PyNotImplementedError, PyValueError},
    types::{PyAnyMethods, PyStringMethods, PyTypeMethods},
};
use pyo3_object_store::PyExternalObjectStore;
use zarrs::storage::{
    ReadableWritableListableStorage, storage_adapter::async_to_sync::AsyncToSyncStorageAdapter,
};

use crate::{runtime::tokio_block_on, utils::PyErrExt};

mod filesystem;
mod http;
mod obstore;

pub use self::filesystem::FilesystemStoreConfig;
pub use self::http::HttpStoreConfig;
pub use self::obstore::ObStoreConfig;

#[derive(Debug, Clone)]
pub enum StoreConfig {
    Filesystem(FilesystemStoreConfig),
    Http(HttpStoreConfig),
    ObStore(ObStoreConfig),
    // TODO: Add support for more stores
}

impl<'py> FromPyObject<'_, 'py> for StoreConfig {
    type Error = PyErr;

    fn extract(store: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
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
            "ObjectStore" => {
                let underlying_store = store.getattr("store")?;
                let external_object_store: PyExternalObjectStore = underlying_store.extract()?;
                let object_store: Arc<dyn zarrs_object_store::object_store::ObjectStore> =
                    external_object_store.into_dyn();
                Ok(StoreConfig::ObStore(ObStoreConfig::new(object_store)))
            }
            _ => Err(PyErr::new::<PyNotImplementedError, _>(format!(
                "zarrs-python does not support {name} stores"
            ))),
        }
    }
}

impl StoreConfig {
    pub fn direct_io(&mut self, flag: bool) -> () {
        match self {
            StoreConfig::Filesystem(config) => config.direct_io(flag),
            StoreConfig::Http(_config) => (),
            StoreConfig::ObStore(_config) => (),
        }
    }
}

impl pyo3_stub_gen::PyStubType for StoreConfig {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("zarr.abc.store.Store", "zarr.abc.store".into())
    }
}

impl TryFrom<&StoreConfig> for ReadableWritableListableStorage {
    type Error = PyErr;

    fn try_from(value: &StoreConfig) -> Result<Self, Self::Error> {
        match value {
            StoreConfig::Filesystem(config) => config.try_into(),
            StoreConfig::Http(config) => config.try_into(),
            StoreConfig::ObStore(config) => config.try_into(),
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
