#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

use chunk_item::WithSubset;
use itertools::Itertools;
use numpy::npyffi::PyArrayObject;
use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use unsafe_cell_slice::UnsafeCellSlice;
use utils::is_whole_chunk;
use zarrs::array::codec::{
    ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, CodecOptions, CodecOptionsBuilder,
    StoragePartialDecoder,
};
use zarrs::array::{
    copy_fill_value_into, update_array_bytes, Array, ArrayBytes, ArrayBytesFixedDisjointView,
    ArrayMetadata, ArraySize, CodecChain, FillValue,
};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableWritableListableStorage, StorageHandle, StoreKey};

mod chunk_item;
mod concurrency;
mod metadata_v2;
mod runtime;
mod store;
#[cfg(test)]
mod tests;
mod utils;

use crate::chunk_item::ChunksItem;
use crate::concurrency::ChunkConcurrentLimitAndCodecOptions;
use crate::metadata_v2::codec_metadata_v2_to_v3;
use crate::store::StoreConfig;
use crate::utils::{PyErrExt as _, PyUntypedArrayExt as _};

// TODO: Use a OnceLock for store with get_or_try_init when stabilised?
#[gen_stub_pyclass]
#[pyclass]
pub struct CodecPipelineImpl {
    pub(crate) store: ReadableWritableListableStorage,
    pub(crate) codec_chain: Arc<CodecChain>,
    pub(crate) codec_options: CodecOptions,
    pub(crate) chunk_concurrent_minimum: usize,
    pub(crate) chunk_concurrent_maximum: usize,
    pub(crate) num_threads: usize,
}

impl CodecPipelineImpl {
    fn retrieve_chunk_bytes<'a, I: ChunksItem>(
        &self,
        item: &I,
        codec_chain: &CodecChain,
        codec_options: &CodecOptions,
    ) -> PyResult<ArrayBytes<'a>> {
        let value_encoded = self.store.get(item.key()).map_py_err::<PyRuntimeError>()?;
        let value_decoded = if let Some(value_encoded) = value_encoded {
            let value_encoded: Vec<u8> = value_encoded.into(); // zero-copy in this case
            codec_chain
                .decode(value_encoded.into(), item.representation(), codec_options)
                .map_py_err::<PyRuntimeError>()?
        } else {
            let array_size = ArraySize::new(
                item.representation().data_type().size(),
                item.representation().num_elements(),
            );
            ArrayBytes::new_fill_value(array_size, item.representation().fill_value())
        };
        Ok(value_decoded)
    }

    fn store_chunk_bytes<I: ChunksItem>(
        &self,
        item: &I,
        codec_chain: &CodecChain,
        value_decoded: ArrayBytes,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        value_decoded
            .validate(
                item.representation().num_elements(),
                item.representation().data_type().size(),
            )
            .map_py_err::<PyValueError>()?;

        if value_decoded.is_fill_value(item.representation().fill_value()) {
            self.store.erase(item.key()).map_py_err::<PyRuntimeError>()
        } else {
            let value_encoded = codec_chain
                .encode(value_decoded, item.representation(), codec_options)
                .map(Cow::into_owned)
                .map_py_err::<PyRuntimeError>()?;

            // Store the encoded chunk
            self.store
                .set(item.key(), value_encoded.into())
                .map_py_err::<PyRuntimeError>()
        }
    }

    fn store_chunk_subset_bytes<I: ChunksItem>(
        &self,
        item: &I,
        codec_chain: &CodecChain,
        chunk_subset_bytes: ArrayBytes,
        chunk_subset: &ArraySubset,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        let array_shape = item.representation().shape_u64();
        if !chunk_subset.inbounds_shape(&array_shape) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "chunk subset ({chunk_subset}) is out of bounds for array shape ({array_shape:?})"
            )));
        }
        let data_type_size = item.representation().data_type().size();

        if chunk_subset.start().iter().all(|&o| o == 0) && chunk_subset.shape() == array_shape {
            // Fast path if the chunk subset spans the entire chunk, no read required
            self.store_chunk_bytes(item, codec_chain, chunk_subset_bytes, codec_options)
        } else {
            // Validate the chunk subset bytes
            chunk_subset_bytes
                .validate(chunk_subset.num_elements(), data_type_size)
                .map_py_err::<PyValueError>()?;

            // Retrieve the chunk
            let chunk_bytes_old = self.retrieve_chunk_bytes(item, codec_chain, codec_options)?;

            // Update the chunk
            let chunk_bytes_new = update_array_bytes(
                chunk_bytes_old,
                &array_shape,
                chunk_subset,
                &chunk_subset_bytes,
                data_type_size,
            )
            .map_py_err::<PyRuntimeError>()?;

            // Store the updated chunk
            self.store_chunk_bytes(item, codec_chain, chunk_bytes_new, codec_options)
        }
    }

    fn py_untyped_array_to_array_object<'a>(
        value: &'a Bound<'_, PyUntypedArray>,
    ) -> &'a PyArrayObject {
        // TODO: Upstream a PyUntypedArray.as_array_ref()?
        //       https://github.com/zarrs/zarrs-python/pull/80/files/75be39184905d688ac04a5f8bca08c5241c458cd#r1918365296
        let array_object_ptr: NonNull<PyArrayObject> = NonNull::new(value.as_array_ptr())
            .expect("bug in numpy crate: Bound<'_, PyUntypedArray>::as_array_ptr unexpectedly returned a null pointer");
        let array_object: &'a PyArrayObject = unsafe {
            // SAFETY: the array object pointed to by array_object_ptr is valid for 'a
            array_object_ptr.as_ref()
        };
        array_object
    }

    fn nparray_to_slice<'a>(value: &'a Bound<'_, PyUntypedArray>) -> Result<&'a [u8], PyErr> {
        if !value.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input array must be a C contiguous array".to_string(),
            ));
        }
        let array_object: &PyArrayObject = Self::py_untyped_array_to_array_object(value);
        let array_data = array_object.data.cast::<u8>();
        let array_len = value.len() * value.dtype().itemsize();
        let slice = unsafe {
            // SAFETY: array_data is a valid pointer to a u8 array of length array_len
            debug_assert!(!array_data.is_null());
            std::slice::from_raw_parts(array_data, array_len)
        };
        Ok(slice)
    }

    fn nparray_to_unsafe_cell_slice<'a>(
        value: &'a Bound<'_, PyUntypedArray>,
    ) -> Result<UnsafeCellSlice<'a, u8>, PyErr> {
        if !value.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input array must be a C contiguous array".to_string(),
            ));
        }
        let array_object: &PyArrayObject = Self::py_untyped_array_to_array_object(value);
        let array_data = array_object.data.cast::<u8>();
        let array_len = value.len() * value.dtype().itemsize();
        let output = unsafe {
            // SAFETY: array_data is a valid pointer to a u8 array of length array_len
            debug_assert!(!array_data.is_null());
            std::slice::from_raw_parts_mut(array_data, array_len)
        };
        Ok(UnsafeCellSlice::new(output))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl CodecPipelineImpl {
    #[pyo3(signature = (
        array_metadata,
        store_config,
        *,
        validate_checksums=None,
        chunk_concurrent_minimum=None,
        chunk_concurrent_maximum=None,
        num_threads=None,
    ))]
    #[new]
    fn new(
        array_metadata: &str,
        store_config: StoreConfig,
        validate_checksums: Option<bool>,
        chunk_concurrent_minimum: Option<usize>,
        chunk_concurrent_maximum: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let metadata: ArrayMetadata =
            serde_json::from_str(array_metadata).map_py_err::<PyTypeError>()?;

        // TODO: Add a direct metadata -> codec chain method to zarrs
        let store = Arc::new(MemoryStore::new());
        let array = Array::new_with_metadata(store, "/", metadata).map_py_err::<PyTypeError>()?;
        let codec_chain = Arc::new(array.codecs().clone());

        let mut codec_options = CodecOptionsBuilder::new();
        if let Some(validate_checksums) = validate_checksums {
            codec_options = codec_options.validate_checksums(validate_checksums);
        }
        let codec_options = codec_options.build();

        let chunk_concurrent_minimum = chunk_concurrent_minimum
            .unwrap_or(zarrs::config::global_config().chunk_concurrent_minimum());
        let chunk_concurrent_maximum =
            chunk_concurrent_maximum.unwrap_or(rayon::current_num_threads());
        let num_threads = num_threads.unwrap_or(rayon::current_num_threads());

        let store: ReadableWritableListableStorage =
            (&store_config).try_into().map_py_err::<PyTypeError>()?;

        Ok(Self {
            store,
            codec_chain,
            codec_options,
            chunk_concurrent_minimum,
            chunk_concurrent_maximum,
            num_threads,
        })
    }

    fn retrieve_chunks_and_apply_index(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::WithSubset>, // FIXME: Ref / iterable?
        value: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        // Get input array
        let output = Self::nparray_to_unsafe_cell_slice(value)?;
        let output_shape: Vec<u64> = value.shape_zarr()?;

        // Adjust the concurrency based on the codec chain and the first chunk description
        let Some((chunk_concurrent_limit, codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return Ok(());
        };

        // Assemble partial decoders ahead of time and in parallel
        let partial_chunk_descriptions = chunk_descriptions
            .iter()
            .filter(|item| !(is_whole_chunk(item)))
            .unique_by(|item| item.key())
            .collect::<Vec<_>>();
        let mut partial_decoder_cache: HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>> =
            HashMap::new();
        if !partial_chunk_descriptions.is_empty() {
            let key_decoder_pairs = iter_concurrent_limit!(
                chunk_concurrent_limit,
                partial_chunk_descriptions,
                map,
                |item| {
                    let storage_handle = Arc::new(StorageHandle::new(self.store.clone()));
                    let input_handle =
                        StoragePartialDecoder::new(storage_handle, item.key().clone());
                    let partial_decoder = self
                        .codec_chain
                        .clone()
                        .partial_decoder(
                            Arc::new(input_handle),
                            item.representation(),
                            &codec_options,
                        )
                        .map_py_err::<PyValueError>()?;
                    Ok((item.key().clone(), partial_decoder))
                }
            )
            .collect::<PyResult<Vec<_>>>()?;
            partial_decoder_cache.extend(key_decoder_pairs);
        }

        py.allow_threads(move || {
            // FIXME: the `decode_into` methods only support fixed length data types.
            // For variable length data types, need a codepath with non `_into` methods.
            // Collect all the subsets and copy into value on the Python side?
            let update_chunk_subset = |item: chunk_item::WithSubset| {
                let chunk_item::WithSubset {
                    item,
                    subset,
                    chunk_subset,
                } = item;
                let mut output_view = unsafe {
                    // TODO: Is the following correct?
                    //       can we guarantee that when this function is called from Python with arbitrary arguments?
                    // SAFETY: chunks represent disjoint array subsets
                    ArrayBytesFixedDisjointView::new(
                        output,
                        // TODO: why is data_type in `item`, it should be derived from `output`, no?
                        item.representation()
                            .data_type()
                            .fixed_size()
                            .ok_or("variable length data type not supported")
                            .map_py_err::<PyTypeError>()?,
                        &output_shape,
                        subset,
                    )
                    .map_py_err::<PyRuntimeError>()?
                };

                // See zarrs::array::Array::retrieve_chunk_subset_into
                if chunk_subset.start().iter().all(|&o| o == 0)
                    && chunk_subset.shape() == item.representation().shape_u64()
                {
                    // See zarrs::array::Array::retrieve_chunk_into
                    if let Some(chunk_encoded) =
                        self.store.get(item.key()).map_py_err::<PyRuntimeError>()?
                    {
                        // Decode the encoded data into the output buffer
                        let chunk_encoded: Vec<u8> = chunk_encoded.into();
                        self.codec_chain.decode_into(
                            Cow::Owned(chunk_encoded),
                            item.representation(),
                            &mut output_view,
                            &codec_options,
                        )
                    } else {
                        // The chunk is missing, write the fill value
                        copy_fill_value_into(
                            item.representation().data_type(),
                            item.representation().fill_value(),
                            &mut output_view,
                        )
                    }
                } else {
                    let key = item.key();
                    let partial_decoder = partial_decoder_cache.get(key).ok_or_else(|| {
                        PyRuntimeError::new_err(format!("Partial decoder not found for key: {key}"))
                    })?;
                    partial_decoder.partial_decode_into(
                        &chunk_subset,
                        &mut output_view,
                        &codec_options,
                    )
                }
                .map_py_err::<PyValueError>()
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

    fn store_chunks_with_indices(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::WithSubset>,
        value: &Bound<'_, PyUntypedArray>,
        write_empty_chunks: bool,
    ) -> PyResult<()> {
        enum InputValue<'a> {
            Array(ArrayBytes<'a>),
            Constant(FillValue),
        }

        // Get input array
        let input_slice = Self::nparray_to_slice(value)?;
        let input = if value.ndim() > 0 {
            // FIXME: Handle variable length data types, convert value to bytes and offsets
            InputValue::Array(ArrayBytes::new_flen(Cow::Borrowed(input_slice)))
        } else {
            InputValue::Constant(FillValue::new(input_slice.to_vec()))
        };
        let input_shape: Vec<u64> = value.shape_zarr()?;

        // Adjust the concurrency based on the codec chain and the first chunk description
        let Some((chunk_concurrent_limit, mut codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return Ok(());
        };
        codec_options.set_store_empty_chunks(write_empty_chunks);

        py.allow_threads(move || {
            let store_chunk = |item: chunk_item::WithSubset| match &input {
                InputValue::Array(input) => {
                    let chunk_subset_bytes = input
                        .extract_array_subset(
                            &item.subset,
                            &input_shape,
                            item.item.representation().data_type(),
                        )
                        .map_py_err::<PyRuntimeError>()?;
                    self.store_chunk_subset_bytes(
                        &item,
                        &self.codec_chain,
                        chunk_subset_bytes,
                        &item.chunk_subset,
                        &codec_options,
                    )
                }
                InputValue::Constant(constant_value) => {
                    let chunk_subset_bytes = ArrayBytes::new_fill_value(
                        ArraySize::new(
                            item.representation().data_type().size(),
                            item.chunk_subset.num_elements(),
                        ),
                        constant_value,
                    );

                    self.store_chunk_subset_bytes(
                        &item,
                        &self.codec_chain,
                        chunk_subset_bytes,
                        &item.chunk_subset,
                        &codec_options,
                    )
                }
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
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<CodecPipelineImpl>()?;
    m.add_class::<chunk_item::Basic>()?;
    m.add_class::<chunk_item::WithSubset>()?;
    m.add_function(wrap_pyfunction!(codec_metadata_v2_to_v3, m)?)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
