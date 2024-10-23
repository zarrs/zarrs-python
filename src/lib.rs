#![warn(clippy::pedantic)]

use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
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
        } else if let Some(chunk_path) = chunk_path.strip_prefix("file://") {
            let store = if self.store.is_none() {
                if chunk_path.starts_with('/') {
                    // Absolute path
                    self.store = Some(CodecPipelineStore::Filesystem(Arc::new(
                        FilesystemStore::new("/").unwrap(),
                    )));
                } else {
                    // Relative path
                    self.store = Some(CodecPipelineStore::Filesystem(Arc::new(
                        FilesystemStore::new(std::env::current_dir().unwrap()).unwrap(),
                    )));
                }
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
            utils::err(format!("unsupported store for {chunk_path}"))
        }
    }

    fn get_chunk_representation(
        chunk_shape: Vec<u64>,
        dtype: &str,
        fill_value: Vec<u8>,
    ) -> PyResult<ChunkRepresentation> {
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
        Ok(chunk_representation)
    }

    fn retrieve_chunk_bytes<'py>(
        store: &dyn ReadableWritableListableStorageTraits,
        key: &StoreKey,
        codec_chain: &CodecChain,
        chunk_representation: &ChunkRepresentation,
    ) -> PyResult<Vec<u8>> {
        let value_encoded = store.get(key).unwrap(); // FIXME: Error handling
        let value_decoded = if let Some(value_encoded) = value_encoded {
            let value_encoded: Vec<u8> = value_encoded.into(); // zero-copy in this case
            codec_chain
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
        Ok(value_decoded)
    }

    fn store_chunk_bytes(
        store: &dyn ReadableWritableListableStorageTraits,
        key: &StoreKey,
        codec_chain: &CodecChain,
        chunk_representation: &ChunkRepresentation,
        value_decoded: ArrayBytes,
    ) -> PyResult<()> {
        let value_encoded = codec_chain
            .encode(
                value_decoded,
                &chunk_representation,
                &CodecOptions::default(),
            )
            .map(Cow::into_owned)
            .unwrap();

        // Store the encoded chunk
        store.set(key, value_encoded.into()).unwrap(); // FIXME: Error handling

        Ok(())
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

    fn retrieve_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_path: &str,
        chunk_shape: Vec<u64>,
        dtype: &str,
        fill_value: Vec<u8>,
    ) -> PyResult<Bound<'py, PyArray<u8, numpy::ndarray::Dim<[usize; 1]>>>> {
        let (store, chunk_path) = self.get_store_and_path(chunk_path)?;
        let key = StoreKey::new(chunk_path).unwrap(); // FIXME: Error handling
        let chunk_representation = Self::get_chunk_representation(chunk_shape, dtype, fill_value)?;

        Ok(Self::retrieve_chunk_bytes(
            store.as_ref(),
            &key,
            &self.codec_chain,
            &chunk_representation,
        )?
        .into_pyarray_bound(py))
    }

    // fn retrieve_chunk_subset<'py>(
    //     &mut self,
    //     py: Python<'py>,
    //     chunk_path: &str,
    //     chunk_shape: Vec<u64>,
    //     dtype: &str,
    //     fill_value: Vec<u8>,
    //     // TODO: Chunk selection
    // ) -> PyResult<Bound<'py, PyArray<u8, numpy::ndarray::Dim<[usize; 1]>>>> {
    //     let (store, chunk_path) = self.get_store_and_path(chunk_path)?;
    //     let key = StoreKey::new(chunk_path).unwrap(); // FIXME: Error handling
    //     let chunk_representation = Self::get_chunk_representation(chunk_shape, dtype, fill_value)?;

    //     // Review zarrs::Array::retrieve_chunk_subset
    //     todo!("retrieve_chunk_subset")
    // }

    fn store_chunk(
        &mut self,
        chunk_path: &str,
        chunk_shape: Vec<u64>,
        dtype: &str,
        fill_value: Vec<u8>,
        value: &Bound<'_, PyBytes>,
    ) -> PyResult<()> {
        let (store, chunk_path) = self.get_store_and_path(chunk_path)?;
        let key = StoreKey::new(chunk_path).unwrap(); // FIXME: Error handling
        let chunk_representation = Self::get_chunk_representation(chunk_shape, dtype, fill_value)?;

        let value_decoded = Cow::Borrowed(value.as_bytes());
        Self::store_chunk_bytes(
            store.as_ref(),
            &key,
            &self.codec_chain,
            &chunk_representation,
            ArrayBytes::new_flen(value_decoded),
        )
    }

    // fn store_chunk_subset(
    //     &mut self,
    //     chunk_path: &str,
    //     chunk_shape: Vec<u64>,
    //     dtype: &str,
    //     fill_value: Vec<u8>,
    //     // out_selection: &Bound<PyTuple>, // FIXME: tuple[Selector, ...] | npt.NDArray[np.intp] | slice
    //     chunk_selection: &Bound<PyTuple>, // FIXME: tuple[Selector, ...] | npt.NDArray[np.intp] | slice
    //     value: &Bound<'_, PyBytes>,
    // ) -> PyResult<()> {
    //     let (store, chunk_path) = self.get_store_and_path(chunk_path)?;
    //     let key = StoreKey::new(chunk_path).unwrap(); // FIXME: Error handling
    //     let chunk_representation = Self::get_chunk_representation(chunk_shape, dtype, fill_value)?;

    //     // Retrieve the chunk
    //     let value_decoded = Self::retrieve_chunk_bytes(
    //         store.as_ref(),
    //         &key,
    //         &self.codec_chain,
    //         &chunk_representation,
    //     )?;

    //     // Update the chunk
    //     let slices = chunk_selection
    //         .iter()
    //         .zip(chunk_representation.shape())
    //         .map(|(selection, shape)| {
    //             // FIXME: BasicSelector | ArrayOfIntOrBool
    //             // FIXME: BasicSelector = int | slice | EllipsisType
    //             // FIXME: ArrayOfIntOrBool = npt.NDArray[np.intp] | npt.NDArray[np.bool_]
    //             let selection = selection.downcast::<PySlice>()?;
    //             selection.indices(shape.get() as i64)
    //         })
    //         .collect::<Result<Vec<_>, _>>()?;
    //     todo!(
    //         "Update the chunk with slices: {:?} from value: {:?}",
    //         slices,
    //         value
    //     );

    //     // Store the updated chunk
    //     Self::store_chunk_bytes(
    //         store.as_ref(),
    //         &key,
    //         &self.codec_chain,
    //         &chunk_representation,
    //         ArrayBytes::new_flen(value_decoded),
    //     )
    // }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CodecPipelineImpl>()?;
    Ok(())
}
