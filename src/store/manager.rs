use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use pyo3::{exceptions::PyRuntimeError, PyResult};
use zarrs::{
    array::codec::StoragePartialDecoder,
    storage::{Bytes, MaybeBytes, ReadableWritableListableStorage, StorageHandle},
};

use crate::{chunk_item::ChunksItem, store::PyErrExt as _};

use super::StoreConfig;

#[derive(Default)]
pub(crate) struct StoreManager(Mutex<BTreeMap<StoreConfig, ReadableWritableListableStorage>>);

impl StoreManager {
    fn store<I: ChunksItem>(&self, item: &I) -> PyResult<ReadableWritableListableStorage> {
        use std::collections::btree_map::Entry::{Occupied, Vacant};
        match self
            .0
            .lock()
            .map_py_err::<PyRuntimeError>()?
            .entry(item.store_config())
        {
            Occupied(e) => Ok(e.get().clone()),
            Vacant(e) => Ok(e.insert((&item.store_config()).try_into()?).clone()),
        }
    }

    pub(crate) fn get<I: ChunksItem>(&self, item: &I) -> PyResult<MaybeBytes> {
        self.store(item)?
            .get(item.key())
            .map_py_err::<PyRuntimeError>()
    }

    pub(crate) fn set<I: ChunksItem>(&self, item: &I, value: Bytes) -> PyResult<()> {
        self.store(item)?
            .set(item.key(), value)
            .map_py_err::<PyRuntimeError>()
    }

    pub(crate) fn erase<I: ChunksItem>(&self, item: &I) -> PyResult<()> {
        self.store(item)?
            .erase(item.key())
            .map_py_err::<PyRuntimeError>()
    }

    pub(crate) fn decoder<I: ChunksItem>(&self, item: &I) -> PyResult<StoragePartialDecoder> {
        // Partially decode the chunk into the output buffer
        let storage_handle = Arc::new(StorageHandle::new(self.store(item)?));
        // NOTE: Normally a storage transformer would exist between the storage handle and the input handle
        // but zarr-python does not support them nor forward them to the codec pipeline
        Ok(StoragePartialDecoder::new(
            storage_handle,
            item.key().clone(),
        ))
    }
}
