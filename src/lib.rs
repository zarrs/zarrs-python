#![warn(clippy::pedantic)]

use pyo3::prelude::*;
use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::codec::{ArrayToBytesCodecTraits, CodecOptions};
use zarrs::array::{
    Array as RustArray, ArrayBytes, ChunkRepresentation, CodecChain, DataType, FillValue,
};
use zarrs::filesystem::FilesystemStore;
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableStorage, ReadableWritableListableStorageTraits, StoreKey};
use zarrs_http::HTTPStore;

mod array;
mod utils;

// #[pyfunction]
// fn open_array(path: &str) -> PyResult<array::ZarrsPythonArray> {
//     #![allow(deprecated)] // HTTPStore is moved to an independent crate in zarrs 0.17 and undeprecated
//     let s: ReadableStorage = if path.starts_with("http://") | path.starts_with("https://") {
//         Arc::new(HTTPStore::new(path).or_else(|x| utils::err(x.to_string()))?)
//     } else {
//         Arc::new(FilesystemStore::new(path).or_else(|x| utils::err(x.to_string()))?)
//     };
//     let arr = RustArray::open(s, "/").or_else(|x| utils::err(x.to_string()))?;
//     Ok(array::ZarrsPythonArray { arr })
// }

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

    fn retrieve_chunk_subset(
        &mut self, // TODO: Interior mut?
        chunk_path: &str,
        // TODO: Chunk selection
        // TODO: Chunk representation
    ) -> PyResult<Vec<u8>> {
        let chunk_representation = ChunkRepresentation::new(
            // FIXME
            vec![NonZeroU64::new(10).unwrap()],
            DataType::UInt8,
            FillValue::from(0u8),
        )
        .unwrap();

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
        Ok(value_decoded)
    }

    fn store_chunk_subset(
        &mut self, // TODO: Interior mut?
        chunk_path: &str,
        // TODO: Chunk selection
        // TODO: Chunk representation
        // TODO: value...
    ) -> PyResult<()> {
        let chunk_representation = ChunkRepresentation::new(
            // FIXME
            vec![NonZeroU64::new(10).unwrap()],
            DataType::UInt8,
            FillValue::from(0u8),
        )
        .unwrap();

        // TODO: Review array.store_chunk_subset
        let (store, chunk_path) = self.get_store_and_path(chunk_path);
        let key = StoreKey::new(chunk_path).unwrap(); // FIXME: Error handling

        let value_decoded = vec![42u8; 10]; // FIXME

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

    // fn encode_chunk(&self, decoded_chunk: &[u8]) -> PyResult<Vec<u8>> {
    //     let decoded_representation: ChunkRepresentation = todo!("pass the chunk representation");
    //     let options = CodecOptions::default();
    //     let bytes = ArrayBytes::new_flen(decoded_chunk);
    //     let encoded = self
    //         .codec_chain
    //         .encode(bytes, &decoded_representation, &options)
    //         .or_else(|x| utils::err(x.to_string()))?;
    //     Ok(encoded.into_owned())
    // }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CodecPipelineImpl>()?;
    // m.add_function(wrap_pyfunction!(open_array, m)?)?;
    // m.add_class::<array::ZarrsPythonArray>()?;
    Ok(())
}
