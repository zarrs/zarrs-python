use std::sync::Arc;

use pyo3::PyErr;
use zarrs::storage::AsyncReadableWritableListableStorage;
use zarrs_object_store::{AsyncObjectStore, object_store::ObjectStore};

#[derive(Debug, Clone)]
pub struct ObStoreConfig {
    store: Arc<dyn ObjectStore>,
}

impl ObStoreConfig {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self { store }
    }
}

impl TryInto<AsyncReadableWritableListableStorage> for &ObStoreConfig {
    type Error = PyErr;

    fn try_into(self) -> Result<AsyncReadableWritableListableStorage, Self::Error> {
        let async_store = Arc::new(AsyncObjectStore::new(self.store.clone()));
        Ok(async_store)
    }
}
