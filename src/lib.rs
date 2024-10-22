#![warn(clippy::pedantic)]

use numpy::{IntoPyArray, PyArray, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::codec::{ArrayToBytesCodecTraits, CodecOptions};
use zarrs::array::{ArrayBytes, ArraySize, ChunkRepresentation, CodecChain, DataType, FillValue};
use zarrs::filesystem::FilesystemStore;
use zarrs::metadata::v3::array::data_type::DataTypeMetadataV3;
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableWritableListableStorageTraits, StoreKey};

mod utils;

pub enum CodecPipelineStore {
    Memory(Arc<MemoryStore>),
    Filesystem(Arc<FilesystemStore>),
}

#[pyclass]
pub struct CodecPipelineImpl {
    pub codec_chain: CodecChain,
    pub store: Option<CodecPipelineStore>,
}

impl CodecPipelineImpl {
    fn get_store_and_path<'a>(
        &mut self,
        chunk_path: &'a str,
    ) -> PyResult<(Arc<dyn ReadableWritableListableStorageTraits>, &'a str)> {
        if let Some(chunk_path) = chunk_path.strip_prefix("memory://") {
            let store = if self.store.is_none() {
                self.store = Some(CodecPipelineStore::Memory(Arc::new(MemoryStore::default())));
                let Some(CodecPipelineStore::Memory(store)) = self.store.as_ref() else {
                    unreachable!()
                };
                store
            } else if let Some(CodecPipelineStore::Memory(store)) = &self.store {
                store
            } else {
                utils::err("the store type changed".to_string())?
            };
            Ok((store.clone(), chunk_path))
        } else if let Some(chunk_path) = chunk_path.strip_prefix("file:///") {
            // NOTE: Extra / is intentional, then chunk_path is relative to root
            let store = if self.store.is_none() {
                self.store = Some(CodecPipelineStore::Filesystem(Arc::new(
                    FilesystemStore::new("/").unwrap(),
                )));
                let Some(CodecPipelineStore::Filesystem(store)) = self.store.as_ref() else {
                    unreachable!()
                };
                store
            } else if let Some(CodecPipelineStore::Filesystem(store)) = &self.store {
                store
            } else {
                utils::err("the store type changed".to_string())?
            };
            Ok((store.clone(), chunk_path))
        } else {
            // TODO: Add support for more stores
            utils::err("unsupported store".to_string())
        }
    }
}

#[pymethods]
impl CodecPipelineImpl {
    #[new]
    fn new(metadata: &str) -> PyResult<Self> {
        let metadata: Vec<MetadataV3> =
            serde_json::from_str(metadata).or_else(|x| utils::err(x.to_string()))?;
        let codec_chain =
            CodecChain::from_metadata(&metadata).or_else(|x| utils::err(x.to_string()))?;
        Ok(Self {
            codec_chain,
            store: None,
        })
    }

    fn retrieve_chunk_subset<'py>(
        &mut self, // TODO: Interior mut?
        py: Python<'py>,
        chunk_path: &str,
        chunk_shape: Vec<u64>,
        dtype: &str,
        fill_value: Vec<u8>,
        // TODO: Chunk selection
    ) -> PyResult<Bound<'py, PyArray<u8, numpy::ndarray::Dim<[usize; 1]>>>> {
        // Get the store and chunk key
        let (store, chunk_path) = self.get_store_and_path(chunk_path)?;
        let key = StoreKey::new(chunk_path).unwrap(); // FIXME: Error handling

        // Check if the entire chunk is being stored
        let is_entire_chunk = true; // FIXME

        // Get the chunk representation
        let data_type =
            DataType::from_metadata(&DataTypeMetadataV3::from_metadata(&MetadataV3::new(dtype)))
                .unwrap(); // yikes
        let chunk_shape = chunk_shape
            .into_iter()
            .map(|x| NonZeroU64::new(x).unwrap())
            .collect();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape, data_type, FillValue::new(fill_value)).unwrap();

        if is_entire_chunk {
            let value_encoded = store.get(&key).unwrap(); // FIXME: Error handling
            let value_decoded = if let Some(value_encoded) = value_encoded {
                let value_encoded: Vec<u8> = value_encoded.into(); // zero-copy in this case
                self.codec_chain
                    .decode(
                        value_encoded.into(),
                        &chunk_representation,
                        &CodecOptions::default(),
                    )
                    .unwrap() // FIXME: Error handling
            } else {
                let array_size = ArraySize::new(
                    chunk_representation.data_type().size(),
                    chunk_representation.num_elements(),
                );
                ArrayBytes::new_fill_value(array_size, chunk_representation.fill_value())
            };
            let value_decoded = value_decoded
                .into_owned()
                .into_fixed()
                .expect("zarrs-python and zarr only support fixed length types")
                .into_owned();
            Ok(value_decoded.into_pyarray_bound(py))
        } else {
            // Review zarrs::Array::retrieve_chunk_subset
            todo!("retrieve_chunk_subset partial chunk")
        }
    }

    fn store_chunk_subset(
        &mut self, // TODO: Interior mut?
        chunk_path: &str,
        chunk_shape: Vec<u64>,
        dtype: &str,
        fill_value: Vec<u8>,
        // TODO: Chunk selection
        value: &Bound<'_, PyArray<u8, numpy::ndarray::IxDyn>>,
    ) -> PyResult<()> {
        // Get the store and chunk key
        let (store, chunk_path) = self.get_store_and_path(chunk_path)?;
        let key = StoreKey::new(chunk_path).unwrap(); // FIXME: Error handling

        // Check if the entire chunk is being stored
        let is_entire_chunk = value
            .shape()
            .iter()
            .zip(chunk_shape.as_slice())
            .map(|(sv, sc)| *sv as u64 == *sc)
            .all(|x| x);

        // Get the chunk representation
        let data_type =
            DataType::from_metadata(&DataTypeMetadataV3::from_metadata(&MetadataV3::new(dtype)))
                .unwrap(); // yikes
        let chunk_shape = chunk_shape
            .into_iter()
            .map(|x| NonZeroU64::new(x).unwrap())
            .collect();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape, data_type, FillValue::new(fill_value)).unwrap();

        if is_entire_chunk {
            // Get the decoded bytes
            let array = unsafe { value.as_array() };
            let value_decoded = if let Some(value_decoded) = array.to_slice() {
                Cow::Borrowed(value_decoded)
            } else {
                Cow::Owned(array.as_standard_layout().into_owned().into_raw_vec())
            };

            let value_encoded = self
                .codec_chain
                .encode(
                    ArrayBytes::new_flen(value_decoded),
                    &chunk_representation,
                    &CodecOptions::default(),
                )
                .map(Cow::into_owned)
                .unwrap();

            // Store the encoded chunk
            store.set(&key, value_encoded.into()).unwrap(); // FIXME: Error handling

            Ok(())
        } else {
            // TODO: Review zarrs::Array::store_chunk_subset
            // Need to retrieve te chunk, update it, and store it
            todo!("store_chunk_subset partial chunk")
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CodecPipelineImpl>()?;
    Ok(())
}
