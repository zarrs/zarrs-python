use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use pyo3::{exceptions::PyRuntimeError, PyResult};
use zarrs::{
    array::{
        codec::{AsyncStoragePartialDecoder, StoragePartialDecoder},
        DataTypeSize,
    },
    storage::{
        AsyncReadableWritableListableStorage, Bytes, MaybeBytes, ReadableWritableListableStorage,
        StorageHandle,
    },
};

use crate::{chunk_item::ChunksItem, store::PyErrExt as _, WithSubset};

use super::StoreConfig;

#[derive(Default)]
pub(crate) struct StoreManager(Mutex<BTreeMap<StoreConfig, ReadableWritableListableStorage>>);

#[derive(Default)]
pub(crate) struct AsyncStoreManager(
    Mutex<BTreeMap<StoreConfig, AsyncReadableWritableListableStorage>>,
);

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

impl AsyncStoreManager {
    fn store<I: ChunksItem>(&self, item: &I) -> PyResult<AsyncReadableWritableListableStorage> {
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

    pub(crate) async fn get<I: ChunksItem>(&self, item: &I) -> PyResult<Option<Bytes>> {
        self.store(item)?
            .get(item.key())
            .await
            .map_py_err::<PyRuntimeError>()
    }

    pub(crate) async fn get_partial_values_key(
        &self,
        item: &WithSubset,
    ) -> PyResult<Option<Vec<Bytes>>> {
        let rep = item.item.representation();
        let shape = rep.shape().iter().map(|i| i.get()).collect::<Vec<_>>();
        let size = match rep.element_size() {
            DataTypeSize::Fixed(size) => Ok(size),
            DataTypeSize::Variable => Err(PyRuntimeError::new_err(
                "Variable length data types not supported in zarrs-python",
            )),
        }?;
        self.store(item)?
            .get_partial_values_key(
                item.key(),
                &item
                    .chunk_subset
                    .byte_ranges(&shape, size)
                    .map_py_err::<PyRuntimeError>()?,
            )
            .await
            .map_py_err::<PyRuntimeError>()
    }

    pub(crate) async fn set<I: ChunksItem>(&self, item: &I, value: Bytes) -> PyResult<()> {
        self.store(item)?
            .set(item.key(), value)
            .await
            .map_py_err::<PyRuntimeError>()
    }

    pub(crate) async fn erase<I: ChunksItem>(&self, item: &I) -> PyResult<()> {
        self.store(item)?
            .erase(item.key())
            .await
            .map_py_err::<PyRuntimeError>()
    }

    pub(crate) fn decoder<I: ChunksItem>(&self, item: &I) -> PyResult<AsyncStoragePartialDecoder> {
        // Partially decode the chunk into the output buffer
        let storage_handle = Arc::new(StorageHandle::new(self.store(item)?));
        // NOTE: Normally a storage transformer would exist between the storage handle and the input handle
        // but zarr-python does not support them nor forward them to the codec pipeline
        Ok(AsyncStoragePartialDecoder::new(
            storage_handle,
            item.key().clone(),
        ))
    }
}
