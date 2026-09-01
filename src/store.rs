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
pub struct StoreConfig {
    pub kind: StoreKind,
    /// Whether zarr-python opened the store read-only, i.e. `mode="r"`.
    ///
    /// zarr-python resolves the mode into the store itself (`Store.read_only`), so it comes
    /// off the same object every other field here does.
    pub read_only: bool,
}

#[derive(Debug, Clone)]
pub enum StoreKind {
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
        let kind = match name {
            "LocalStore" => {
                let root: String = store.getattr("root")?.call_method0("__str__")?.extract()?;
                StoreKind::Filesystem(FilesystemStoreConfig::new(root))
            }
            "FsspecStore" => {
                let fs = store.getattr("fs")?;
                let fs_name = fs.get_type().name()?;
                let fs_name = fs_name.to_str()?;
                let path: String = store.getattr("path")?.extract()?;
                let storage_options: HashMap<String, Bound<'py, PyAny>> =
                    fs.getattr("storage_options")?.extract()?;
                match fs_name {
                    "HTTPFileSystem" => {
                        StoreKind::Http(HttpStoreConfig::new(&path, &storage_options)?)
                    }
                    _ => {
                        return Err(PyErr::new::<PyNotImplementedError, _>(format!(
                            "zarrs-python does not support {fs_name} (FsspecStore) stores"
                        )));
                    }
                }
            }
            "ObjectStore" => {
                let underlying_store = store.getattr("store")?;
                let external_object_store: PyExternalObjectStore = underlying_store.extract()?;
                let object_store: Arc<dyn zarrs_object_store::object_store::ObjectStore> =
                    external_object_store.into_dyn();
                StoreKind::ObStore(ObStoreConfig::new(object_store))
            }
            _ => {
                return Err(PyErr::new::<PyNotImplementedError, _>(format!(
                    "zarrs-python does not support {name} stores"
                )));
            }
        };
        Ok(StoreConfig {
            kind,
            read_only: store.getattr("read_only")?.extract()?,
        })
    }
}

impl StoreConfig {
    pub fn direct_io(&mut self, flag: bool) {
        match &mut self.kind {
            StoreKind::Filesystem(config) => config.direct_io(flag),
            StoreKind::Http(_config) => (),
            StoreKind::ObStore(_config) => (),
        }
    }

    pub fn file_handle_cache_size(&mut self, size: usize) {
        match &mut self.kind {
            StoreKind::Filesystem(config) => config.file_handle_cache_size(size),
            StoreKind::Http(_config) => (),
            StoreKind::ObStore(_config) => (),
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
        match &value.kind {
            StoreKind::Filesystem(config) => config.try_into(),
            StoreKind::Http(config) => config.try_into(),
            StoreKind::ObStore(config) => config.try_into(),
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
