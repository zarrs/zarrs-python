use std::sync::Arc;

use pyo3::PyErr;
use zarrs::storage::{
    ReadableWritableListableStorage, storage_adapter::async_to_sync::AsyncToSyncStorageAdapter,
};
use zarrs_object_store::{AsyncObjectStore, object_store::ObjectStore};

use crate::runtime::tokio_block_on;

#[derive(Debug, Clone)]
pub struct ObStoreConfig {
    store: Arc<dyn ObjectStore>,
}

impl ObStoreConfig {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self { store }
    }
}

impl TryInto<ReadableWritableListableStorage> for &ObStoreConfig {
    type Error = PyErr;

    fn try_into(self) -> Result<ReadableWritableListableStorage, Self::Error> {
        let async_store = Arc::new(AsyncObjectStore::new(self.store.clone()));
        let sync_store = Arc::new(AsyncToSyncStorageAdapter::new(
            async_store,
            tokio_block_on(),
        ));
        Ok(sync_store)
    }
}
