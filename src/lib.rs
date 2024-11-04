#![warn(clippy::pedantic)]

use numpy::npyffi::PyArrayObject;
use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::{
    ArrayToBytesCodecTraits, CodecOptions, CodecOptionsBuilder, StoragePartialDecoder,
};
use zarrs::array::{
    copy_fill_value_into, update_array_bytes, ArrayBytes, ArraySize, ChunkRepresentation,
    CodecChain, DataType, FillValue,
};
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::FilesystemStore;
use zarrs::metadata::v3::array::data_type::DataTypeMetadataV3;
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::{ReadableWritableListableStorageTraits, StorageHandle, StoreKey};

mod utils;

pub enum CodecPipelineStore {
    Filesystem(Arc<FilesystemStore>),
}

#[pyclass]
pub struct CodecPipelineImpl {
    pub codec_chain: Arc<CodecChain>,
    pub store: Arc<Mutex<Option<CodecPipelineStore>>>,
    codec_options: CodecOptions,
}

impl CodecPipelineImpl {
    fn get_store_and_path<'a>(
        &self,
        chunk_path: &'a str,
    ) -> PyResult<(Arc<dyn ReadableWritableListableStorageTraits>, &'a str)> {
        let mut gstore = self.store.lock().unwrap();
        if let Some(chunk_path) = chunk_path.strip_prefix("file://") {
            if gstore.is_none() {
                if let Some(chunk_path) = chunk_path.strip_prefix('/') {
                    // Absolute path
                    let store = Arc::new(
                        FilesystemStore::new("/")
                            .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?,
                    );
                    *gstore = Some(CodecPipelineStore::Filesystem(store.clone()));
                    Ok((store, chunk_path))
                } else {
                    // Relative path
                    let store = Arc::new(
                        FilesystemStore::new(
                            std::env::current_dir()
                                .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?,
                        )
                        .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?,
                    );
                    *gstore = Some(CodecPipelineStore::Filesystem(store.clone()));
                    Ok((store, chunk_path))
                }
            } else if let Some(CodecPipelineStore::Filesystem(store)) = gstore.as_ref() {
                if let Some(chunk_path) = chunk_path.strip_prefix('/') {
                    Ok((store.clone(), chunk_path))
                } else {
                    Ok((store.clone(), chunk_path))
                }
            } else {
                utils::err("the store type changed".to_string())?
            }
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
                .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?;
        let chunk_shape = chunk_shape
            .into_iter()
            .map(|x| NonZeroU64::new(x).expect("chunk shapes should always be non-zero"))
            .collect();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape, data_type, FillValue::new(fill_value))
                .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
        Ok(chunk_representation)
    }

    fn retrieve_chunk_bytes(
        store: &dyn ReadableWritableListableStorageTraits,
        key: &StoreKey,
        codec_chain: &CodecChain,
        chunk_representation: &ChunkRepresentation,
        codec_options: &CodecOptions,
    ) -> PyResult<Vec<u8>> {
        let value_encoded = store
            .get(key)
            .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?;
        let value_decoded = if let Some(value_encoded) = value_encoded {
            let value_encoded: Vec<u8> = value_encoded.into(); // zero-copy in this case
            codec_chain
                .decode(value_encoded.into(), chunk_representation, codec_options)
                .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?
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
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        if value_decoded.is_fill_value(&chunk_representation.fill_value()) {
            store.erase(&key)
        } else {
            let value_encoded = codec_chain
                .encode(value_decoded, chunk_representation, codec_options)
                .map(Cow::into_owned)
                .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?;

            // Store the encoded chunk
            store.set(key, value_encoded.into())
        }
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))
    }

    fn store_chunk_subset_bytes(
        store: &dyn ReadableWritableListableStorageTraits,
        key: &StoreKey,
        codec_chain: &CodecChain,
        chunk_representation: &ChunkRepresentation,
        chunk_subset_bytes: &ArrayBytes,
        chunk_subset: &ArraySubset,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        // Retrieve the chunk
        let chunk_bytes_old = Self::retrieve_chunk_bytes(
            store,
            key,
            codec_chain,
            chunk_representation,
            codec_options,
        )?;

        // Update the chunk
        let chunk_bytes_new = unsafe {
            update_array_bytes(
                ArrayBytes::new_flen(chunk_bytes_old),
                &chunk_representation.shape_u64(),
                chunk_subset,
                chunk_subset_bytes,
                chunk_representation.data_type().size(),
            )
        };

        // Store the updated chunk
        Self::store_chunk_bytes(
            store,
            key,
            codec_chain,
            chunk_representation,
            chunk_bytes_new,
            codec_options,
        )
    }

    fn selection_to_array_subset(selection: &[&PySlice], shape: &[u64]) -> PyResult<ArraySubset> {
        let chunk_ranges = selection
            .iter()
            .zip(shape)
            .map(|(selection, &shape)| {
                let indices = selection.indices(i64::try_from(shape).unwrap())?;
                assert!(indices.start >= 0); // FIXME
                assert!(indices.stop >= 0); // FIXME
                assert!(indices.step == 1);
                let start = u64::try_from(indices.start).unwrap();
                let stop = u64::try_from(indices.stop).unwrap();
                Ok(start..stop)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(ArraySubset::new_with_ranges(&chunk_ranges))
    }

    fn nparray_to_slice<'a>(value: &'a Bound<'_, PyUntypedArray>) -> &'a [u8] {
        // TODO: is this and the below a bug? why doesn't .itemsize() work?
        let itemsize = value
            .dtype()
            .getattr("itemsize")
            .unwrap()
            .extract::<usize>()
            .unwrap();
        let array_object_ptr: *mut PyArrayObject = value.as_array_ptr();
        let array_object: &mut PyArrayObject = unsafe { array_object_ptr.as_mut().unwrap() };
        let array_data = array_object.data.cast::<u8>();
        let array_len = value.len() * itemsize;
        let slice = unsafe { std::slice::from_raw_parts(array_data, array_len) };
        slice
    }

    fn nparray_to_unsafe_cell_slice<'a>(
        value: &'a Bound<'_, PyUntypedArray>,
    ) -> UnsafeCellSlice<'a, u8> {
        let itemsize = value
            .dtype()
            .getattr("itemsize")
            .unwrap()
            .extract::<usize>()
            .unwrap();
        let array_object_ptr: *mut PyArrayObject = value.as_array_ptr();
        let array_object: &mut PyArrayObject = unsafe { array_object_ptr.as_mut().unwrap() };
        let array_data = array_object.data.cast::<u8>();
        let array_len = value.len() * itemsize;
        let output = unsafe { std::slice::from_raw_parts_mut(array_data, array_len) };
        UnsafeCellSlice::new(output)
    }
}

type RetrieveChunksItem<'a> = (
    String,
    Vec<u64>,
    String,
    Vec<u8>,
    Vec<&'a PySlice>,
    Vec<&'a PySlice>,
);

type StoreChunksItem<'a> = (
    String,
    Vec<u64>,
    String,
    Vec<u8>,
    Vec<&'a PySlice>,
    Vec<&'a PySlice>,
);

#[pymethods]
impl CodecPipelineImpl {
    #[new]
    fn new(
        metadata: &str,
        validate_checksums: Option<bool>,
        store_empty_chunks: Option<bool>,
        concurrent_target: Option<usize>,
    ) -> PyResult<Self> {
        let metadata: Vec<MetadataV3> =
            serde_json::from_str(metadata).or_else(|x| utils::err(x.to_string()))?;
        let codec_chain =
            Arc::new(CodecChain::from_metadata(&metadata).or_else(|x| utils::err(x.to_string()))?);
        let mut codec_options = CodecOptionsBuilder::new();
        if let Some(validate_checksums) = validate_checksums {
            codec_options = codec_options.validate_checksums(validate_checksums);
        }
        if let Some(store_empty_chunks) = store_empty_chunks {
            codec_options = codec_options.store_empty_chunks(store_empty_chunks);
        }
        if let Some(concurrent_target) = concurrent_target {
            codec_options = codec_options.concurrent_target(concurrent_target);
        }
        let codec_options = codec_options.build();

        Ok(Self {
            codec_chain,
            store: Arc::new(Mutex::new(None)),
            codec_options,
        })
    }

    fn retrieve_chunks(
        &self,
        py: Python,
        chunk_descriptions: Vec<RetrieveChunksItem>, // FIXME: Ref / iterable?
        value: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        // Get input array
        if !value.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input array must be a C contiguous array".to_string(),
            ));
        }
        let output = Self::nparray_to_unsafe_cell_slice(value);
        let output_shape: Vec<u64> = value.shape().iter().map(|&i| i as u64).collect();

        // Collect chunk descriptions
        let chunk_descriptions = chunk_descriptions
            .into_iter()
            .map(
                |(chunk_path, chunk_shape, dtype, fill_value, out_selection, chunk_selection)| {
                    let (store, path) = self.get_store_and_path(&chunk_path)?;
                    Ok((
                        store,
                        path.to_string(),
                        Self::selection_to_array_subset(&chunk_selection, &chunk_shape)?,
                        Self::selection_to_array_subset(&out_selection, &output_shape)?,
                        Self::get_chunk_representation(chunk_shape, &dtype, fill_value)?,
                    ))
                },
            )
            .collect::<PyResult<Vec<_>>>()?;

        py.allow_threads(move || {
            // FIXME: Proper chunk/codec concurrency setup
            let chunk_concurrent_limit = self.codec_options.concurrent_target();
            let codec_options = &self.codec_options;

            let update_chunk_subset =
                |(store, chunk_path, chunk_subset, output_subset, chunk_representation): (
                    Arc<dyn ReadableWritableListableStorageTraits>,
                    String,
                    ArraySubset,
                    ArraySubset,
                    ChunkRepresentation,
                )| {
                    let key = StoreKey::new(chunk_path)
                        .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;

                    // See zarrs::array::Array::retrieve_chunk_subset_into
                    if chunk_subset.start().iter().all(|&o| o == 0)
                        && chunk_subset.shape() == chunk_representation.shape_u64()
                    {
                        // See zarrs::array::Array::retrieve_chunk_into
                        let chunk_encoded = store
                            .get(&key)
                            .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?;
                        if let Some(chunk_encoded) = chunk_encoded {
                            // Decode the encoded data into the output buffer
                            let chunk_encoded: Vec<u8> = chunk_encoded.into();
                            unsafe {
                                self.codec_chain.decode_into(
                                    Cow::Owned(chunk_encoded),
                                    &chunk_representation,
                                    &output,
                                    &output_shape,
                                    &output_subset,
                                    codec_options,
                                )
                            }
                        } else {
                            // The chunk is missing, write the fill value
                            unsafe {
                                copy_fill_value_into(
                                    chunk_representation.data_type(),
                                    chunk_representation.fill_value(),
                                    &output,
                                    &output_shape,
                                    &output_subset,
                                )
                            }
                        }
                    } else {
                        // Partially decode the chunk into the output buffer
                        let storage_handle = Arc::new(StorageHandle::new(store.clone()));
                        // NOTE: Normally a storage transformer would exist between the storage handle and the input handle
                        // but zarr-python does not support them nor forward them to the codec pipeline
                        let input_handle =
                            Arc::new(StoragePartialDecoder::new(storage_handle, key));
                        let partial_decoder = self
                            .codec_chain
                            .clone()
                            .partial_decoder(input_handle, &chunk_representation, codec_options)
                            .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
                        unsafe {
                            partial_decoder.partial_decode_into(
                                &chunk_subset,
                                &output,
                                &output_shape,
                                &output_subset,
                                codec_options,
                            )
                        }
                    }
                    .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))
                };

            iter_concurrent_limit!(
                chunk_concurrent_limit,
                chunk_descriptions,
                try_for_each,
                update_chunk_subset
            )?;

            Ok(())
        })
    }

    fn store_chunks(
        &self,
        py: Python,
        chunk_descriptions: Vec<StoreChunksItem>,
        value: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        // Get input array
        if !value.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input array must be a C contiguous array".to_string(),
            ));
        }
        let input_slice = Self::nparray_to_slice(value);
        let input = ArrayBytes::new_flen(Cow::Borrowed(input_slice));
        let input_shape: Vec<u64> = value.shape().iter().map(|&i| i as u64).collect();

        // Collect chunk descriptions
        let chunk_descriptions = chunk_descriptions
            .into_iter()
            .map(
                |(chunk_path, chunk_shape, dtype, fill_value, out_selection, chunk_selection)| {
                    let (store, path) = self.get_store_and_path(&chunk_path)?;
                    let key = StoreKey::new(path)
                        .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
                    Ok((
                        store,
                        key,
                        Self::selection_to_array_subset(&chunk_selection, &chunk_shape)?,
                        Self::selection_to_array_subset(&out_selection, &input_shape)?,
                        Self::get_chunk_representation(chunk_shape, &dtype, fill_value)?,
                    ))
                },
            )
            .collect::<PyResult<Vec<_>>>()?;

        py.allow_threads(move || {
            // FIXME: Proper chunk/codec concurrency setup
            let chunk_concurrent_limit = self.codec_options.concurrent_target();
            let codec_options = &self.codec_options;

            let store_chunk = |(store, key, chunk_subset, input_subset, chunk_representation): (
                Arc<dyn ReadableWritableListableStorageTraits>,
                StoreKey,
                ArraySubset,
                ArraySubset,
                ChunkRepresentation,
            )| {
                let chunk_subset_bytes = if input_subset.dimensionality() == 0 {
                    // Fast path for setting entire chunks to the fill value
                    let is_entire_chunk = input_subset.start().iter().all(|&o| o == 0)
                        && input_subset.shape() == chunk_representation.shape_u64();
                    if is_entire_chunk
                        && input_slice.to_vec() == chunk_representation.fill_value().as_ne_bytes()
                    {
                        return store
                            .erase(&key)
                            .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()));
                    }

                    // The input is a constant value
                    ArrayBytes::new_fill_value(
                        ArraySize::new(
                            chunk_representation.data_type().size(),
                            chunk_representation.num_elements(),
                        ),
                        &FillValue::new(input_slice.to_vec()),
                    )
                } else {
                    input
                        .extract_array_subset(
                            &input_subset,
                            &input_shape,
                            chunk_representation.data_type(),
                        )
                        .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?
                };

                Self::store_chunk_subset_bytes(
                    store.as_ref(),
                    &key,
                    &self.codec_chain,
                    &chunk_representation,
                    &chunk_subset_bytes,
                    &chunk_subset,
                    codec_options,
                )
            };

            iter_concurrent_limit!(
                chunk_concurrent_limit,
                chunk_descriptions,
                try_for_each,
                store_chunk
            )?;

            Ok(())
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CodecPipelineImpl>()?;
    Ok(())
}
