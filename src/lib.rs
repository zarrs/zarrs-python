#![warn(clippy::pedantic)]

use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::codec::{ArrayToBytesCodecTraits, CodecOptions};
use zarrs::array::{ArrayBytes, ChunkRepresentation, CodecChain, DataType, FillValue};
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
    ) -> (Arc<dyn ReadableWritableListableStorageTraits>, &'a str) {
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
                panic!()
            };
            (store.clone(), chunk_path)
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
                panic!()
            };
            (store.clone(), chunk_path)
        } else {
            todo!("raise error unsupported store")
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
        let data_type =
            DataType::from_metadata(&DataTypeMetadataV3::from_metadata(&MetadataV3::new(dtype)))
                .unwrap(); // yikes
        let chunk_shape = chunk_shape
            .into_iter()
            .map(|x| NonZeroU64::new(x).unwrap())
            .collect();

        let chunk_representation =
            ChunkRepresentation::new(chunk_shape, data_type, FillValue::new(fill_value)).unwrap();

        let (store, chunk_path) = self.get_store_and_path(chunk_path);
        let key = StoreKey::new(chunk_path).unwrap(); // FIXME: Error handling

        // TODO: Use partial decoder, rather than getting all bytes, see Array::retrieve_chunk_subset
        let value_encoded = store.get(&key).unwrap(); // FIXME: Error handling
                                                      // TODO: Decode the value
        let value_encoded: Vec<u8> = value_encoded.unwrap().into();

        let value_decoded = self
            .codec_chain
            .decode(
                value_encoded.into(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .map(|x| x.into_owned())
            .unwrap()
            .into_fixed()
            .unwrap()
            .into_owned();
        Ok(value_decoded.into_pyarray_bound(py))
    }

    fn store_chunk_subset(
        &mut self, // TODO: Interior mut?
        chunk_path: &str,
        chunk_shape: Vec<u64>,
        dtype: &str,
        fill_value: Vec<u8>,
        // TODO: Chunk selection
        // TODO: value...
    ) -> PyResult<()> {
        let value_decoded = // FIXME
            vec![
                42u8;
                usize::try_from(chunk_shape.iter().product::<u64>()).unwrap()
            ];
        let data_type =
            DataType::from_metadata(&DataTypeMetadataV3::from_metadata(&MetadataV3::new(dtype)))
                .unwrap(); // yikes
        let chunk_shape = chunk_shape
            .into_iter()
            .map(|x| NonZeroU64::new(x).unwrap())
            .collect();
        let chunk_representation = ChunkRepresentation::new(
            // FIXME
            chunk_shape,
            data_type,
            FillValue::new(fill_value),
        )
        .unwrap();

        // TODO: Review array.store_chunk_subset
        let (store, chunk_path) = self.get_store_and_path(chunk_path);
        let key = StoreKey::new(chunk_path).unwrap(); // FIXME: Error handling

        let value_encoded = self
            .codec_chain
            .encode(
                ArrayBytes::new_flen(&value_decoded),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .map(|x| x.into_owned())
            .unwrap();

        // Store the encoded chunk
        store.set(&key, value_encoded.into()).unwrap(); // FIXME: Error handling
        Ok(())
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CodecPipelineImpl>()?;
    Ok(())
}
