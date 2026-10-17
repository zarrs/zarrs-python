//! Asynchronous codec pipeline.
//!
//! [`AsyncCodecPipelineImpl`] mirrors [`crate::CodecPipelineImpl`], but holds an
//! [`AsyncReadableWritableListableStorage`] and talks to it with `async`
//! `get`/`set`/`erase` calls.
//!
//! Its methods return Python awaitables (via [`pyo3_async_runtimes`]) driven on
//! the shared Tokio runtime, so the asyncio event loop is never blocked.
//!
//! Those futures must be `Send + 'static`, so they cannot borrow the output
//! `numpy` buffer. Instead, [`AsyncCodecPipelineImpl::retrieve_chunks`] returns
//! one awaitable per chunk resolving to that chunk's decoded subset, and Python
//! copies each into the output as soon as it arrives. A decoded subset is
//! dropped right after its copy, so peak memory is bounded by the chunks in
//! flight rather than by the whole output.
//!
//! The write path only *reads* the input buffer; its future keeps the array
//! alive via a [`Py`] reference and reads it through a raw pointer.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use futures::stream::{self, TryStreamExt};
use itertools::Itertools;
use numpy::{IntoPyArray, PyArray1, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::sync::OnceCell;
use zarrs::array::{
    ArrayBytes, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits, CodecChain, CodecOptions,
    DataType, FillValue, update_array_bytes,
};
use zarrs::storage::{
    AsyncReadableStorage, AsyncReadableStorageTraits, AsyncReadableWritableListableStorage,
    AsyncWritableStorageTraits, StorageHandle,
};
// Not re-exported by `zarrs` (only the async traits are); see Cargo.toml.
use zarrs_codec::AsyncStoragePartialDecoder;

use crate::chunk_item::ChunkItem;
use crate::concurrency::{ChunkConcurrentLimitAndCodecOptions, CodecPipelineConfig};
use crate::store::StoreConfig;
use crate::utils::{PyCodecErrExt, PyErrExt as _, is_whole_chunk};
use crate::{ArrayConfig, CodecPipelineImpl, parse_array_config};

type PartialDecoderCell = Arc<OnceCell<Arc<dyn AsyncArrayPartialDecoderTraits>>>;

/// A decoded chunk subset, converted to a 1D `uint8` numpy array without copying.
struct DecodedBytes(Vec<u8>);

impl<'py> IntoPyObject<'py> for DecodedBytes {
    type Target = PyArray1<u8>;
    type Output = Bound<'py, PyArray1<u8>>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0.into_pyarray(py))
    }
}

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct AsyncCodecPipelineImpl {
    /// Read handle, deliberately the READ-ONLY type; see [`CodecPipelineImpl`].
    pub(crate) readable_store: AsyncReadableStorage,
    /// The writable handle -- `None` when zarr-python opened the store read-only.
    pub(crate) writable_store: Option<AsyncReadableWritableListableStorage>,
    pub(crate) codec_chain: Arc<CodecChain>,
    pub(crate) codec_options: CodecOptions,
    pub(crate) chunk_concurrent_minimum: usize,
    pub(crate) chunk_concurrent_maximum: usize,
    pub(crate) num_threads: usize,
    pub(crate) fill_value: FillValue,
    pub(crate) data_type: DataType,
}

impl CodecPipelineConfig for AsyncCodecPipelineImpl {
    fn codec_chain(&self) -> &CodecChain {
        &self.codec_chain
    }
    fn data_type(&self) -> &DataType {
        &self.data_type
    }
    fn codec_options(&self) -> &CodecOptions {
        &self.codec_options
    }
    fn chunk_concurrent_minimum(&self) -> usize {
        self.chunk_concurrent_minimum
    }
    fn chunk_concurrent_maximum(&self) -> usize {
        self.chunk_concurrent_maximum
    }
    fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl AsyncCodecPipelineImpl {
    async fn retrieve_chunk_bytes<'a>(
        &self,
        item: &ChunkItem,
        codec_options: &CodecOptions,
    ) -> PyResult<ArrayBytes<'a>> {
        let value_encoded = self
            .readable_store
            .get(&item.key)
            .await
            .map_py_err::<PyRuntimeError>()?;
        let value_decoded = if let Some(value_encoded) = value_encoded {
            let value_encoded: Vec<u8> = value_encoded.into(); // zero-copy in this case
            self.codec_chain
                .decode(
                    value_encoded.into(),
                    &item.shape,
                    &self.data_type,
                    &self.fill_value,
                    codec_options,
                )
                .map_codec_err()?
        } else {
            ArrayBytes::new_fill_value(&self.data_type, item.num_elements, &self.fill_value)
                .map_py_err::<PyRuntimeError>()?
        };
        Ok(value_decoded)
    }

    /// The writable store, or zarr-python's own refusal, verbatim.
    fn writable(&self) -> PyResult<&AsyncReadableWritableListableStorage> {
        self.writable_store.as_ref().ok_or_else(|| {
            PyValueError::new_err("store was opened in read-only mode and does not support writing")
        })
    }

    async fn store_chunk_bytes(
        &self,
        item: &ChunkItem,
        value_decoded: ArrayBytes<'_>,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        value_decoded
            .validate(item.num_elements, &self.data_type)
            .map_codec_err()?;

        let store = self.writable()?;

        if value_decoded.is_fill_value(&self.fill_value) {
            store.erase(&item.key).await.map_py_err::<PyRuntimeError>()
        } else {
            let value_encoded = self
                .codec_chain
                .encode(
                    value_decoded,
                    &item.shape,
                    &self.data_type,
                    &self.fill_value,
                    codec_options,
                )
                .map(Cow::into_owned)
                .map_codec_err()?;

            // Store the encoded chunk
            store
                .set(&item.key, value_encoded.into())
                .await
                .map_py_err::<PyRuntimeError>()
        }
    }

    async fn store_chunk_subset_bytes(
        &self,
        item: &ChunkItem,
        chunk_subset_bytes: ArrayBytes<'_>,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        let array_shape = &item.shape;
        let chunk_subset = &item.chunk_subset;
        if !chunk_subset.inbounds_shape(bytemuck::must_cast_slice(array_shape)) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "chunk subset ({chunk_subset}) is out of bounds for array shape ({array_shape:?})"
            )));
        }
        let data_type_size = self.data_type.size();

        if is_whole_chunk(item) {
            // Fast path if the chunk subset spans the entire chunk, no read required
            self.store_chunk_bytes(item, chunk_subset_bytes, codec_options)
                .await
        } else {
            // Validate the chunk subset bytes
            chunk_subset_bytes
                .validate(chunk_subset.num_elements(), &self.data_type)
                .map_codec_err()?;

            // Retrieve the chunk
            let chunk_bytes_old = self.retrieve_chunk_bytes(item, codec_options).await?;

            // Update the chunk
            let chunk_bytes_new = update_array_bytes(
                chunk_bytes_old,
                bytemuck::must_cast_slice(array_shape),
                chunk_subset,
                &chunk_subset_bytes,
                data_type_size,
            )
            .map_codec_err()?;

            // Store the updated chunk
            self.store_chunk_bytes(item, chunk_bytes_new, codec_options)
                .await
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl AsyncCodecPipelineImpl {
    #[pyo3(signature = (
        array_metadata,
        store_config,
        *,
        validate_checksums=false,
        chunk_concurrent_minimum=None,
        chunk_concurrent_maximum=None,
        num_threads=None,
        direct_io=false,
        file_handle_cache_size=0,
    ))]
    #[new]
    fn new(
        array_metadata: &str,
        mut store_config: StoreConfig,
        validate_checksums: bool,
        chunk_concurrent_minimum: Option<usize>,
        chunk_concurrent_maximum: Option<usize>,
        num_threads: Option<usize>,
        direct_io: bool,
        file_handle_cache_size: usize,
    ) -> PyResult<Self> {
        // `direct_io` and `file_handle_cache_size` only affect the (synchronous)
        // filesystem store, which the async pipeline does not support; kept for
        // signature parity.
        store_config.direct_io(direct_io);
        store_config.file_handle_cache_size(file_handle_cache_size);
        let ArrayConfig {
            codec_chain,
            codec_options,
            chunk_concurrent_minimum,
            chunk_concurrent_maximum,
            num_threads,
            fill_value,
            data_type,
        } = parse_array_config(
            array_metadata,
            validate_checksums,
            chunk_concurrent_minimum,
            chunk_concurrent_maximum,
            num_threads,
        )?;

        let store: AsyncReadableWritableListableStorage =
            (&store_config).try_into().map_py_err::<PyTypeError>()?;
        let writable_store = (!store_config.read_only).then(|| store.clone());
        let readable_store: AsyncReadableStorage = store.readable();

        Ok(Self {
            readable_store,
            writable_store,
            codec_chain,
            codec_options,
            chunk_concurrent_minimum,
            chunk_concurrent_maximum,
            num_threads,
            fill_value,
            data_type,
        })
    }

    /// One awaitable per chunk, each resolving to that chunk's decoded subset as a
    /// flat `uint8` array; the caller copies it into the output.
    #[gen_stub(override_return_type(
        type_repr = "builtins.list[collections.abc.Awaitable[numpy.typing.NDArray[numpy.uint8]]]",
        imports = ("builtins", "collections.abc", "numpy", "numpy.typing")
    ))]
    fn retrieve_chunks<'py>(
        &self,
        py: Python<'py>,
        chunk_descriptions: Vec<ChunkItem>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        // Only the codec options are needed here; chunk-level parallelism is
        // delegated to the tokio runtime rather than a manual concurrency limit.
        let Some((_chunk_concurrent_limit, codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return Ok(vec![]);
        };
        // Cheap (all `Arc`s / small values); the futures must own it.
        let pipeline = Arc::new(self.clone());
        let codec_options = Arc::new(codec_options);

        // Subsets sharing a (partial) chunk share one lazily-built partial decoder.
        let partial_decoders: HashMap<_, PartialDecoderCell> = chunk_descriptions
            .iter()
            .filter(|item| !is_whole_chunk(item))
            .map(|item| item.key.clone())
            .unique()
            .map(|key| (key, Arc::default()))
            .collect();

        chunk_descriptions
            .into_iter()
            .map(|item| {
                let p = pipeline.clone();
                let codec_options = codec_options.clone();
                let partial_decoder = partial_decoders.get(&item.key).cloned();
                pyo3_async_runtimes::tokio::future_into_py(py, async move {
                    let bytes = if let Some(cell) = partial_decoder {
                        let partial_decoder = cell
                            .get_or_try_init(|| async {
                                let input_handle = Arc::new(AsyncStoragePartialDecoder::new(
                                    Arc::new(StorageHandle::new(p.readable_store.clone())),
                                    item.key.clone(),
                                ));
                                p.codec_chain
                                    .clone()
                                    .async_partial_decoder(
                                        input_handle,
                                        &item.shape,
                                        &p.data_type,
                                        &p.fill_value,
                                        &codec_options,
                                    )
                                    .await
                                    .map_codec_err()
                            })
                            .await?;
                        partial_decoder
                            .partial_decode(&item.chunk_subset, &codec_options)
                            .await
                            .map_codec_err()?
                            .into_owned()
                    } else {
                        p.retrieve_chunk_bytes(&item, &codec_options).await?
                    };
                    // FIXME: only supports fixed length data types.
                    let bytes = bytes.into_fixed().map_py_err::<PyTypeError>()?;
                    Ok(DecodedBytes(bytes.into_owned()))
                })
            })
            .collect()
    }

    #[gen_stub(override_return_type(
        type_repr = "collections.abc.Awaitable[None]",
        imports = ("collections.abc")
    ))]
    fn store_chunks_with_indices<'py>(
        &self,
        py: Python<'py>,
        chunk_descriptions: Vec<ChunkItem>,
        value: &Bound<'py, PyUntypedArray>,
        write_empty_chunks: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Fail before decoding anything; the write site checks again by construction.
        self.writable()?;

        enum InputValue {
            /// Address and length of `value`'s buffer, kept alive by `value_ref`.
            Array(usize, usize),
            Constant(FillValue),
        }

        // Get input array
        let input_slice = CodecPipelineImpl::nparray_to_slice(value)?;
        let input = if value.ndim() > 0 {
            InputValue::Array(input_slice.as_ptr() as usize, input_slice.len())
        } else {
            InputValue::Constant(FillValue::new(input_slice.to_vec()))
        };
        let value_ref = value.clone().unbind();

        // Adjust the concurrency based on the codec chain and the first chunk description
        let Some((chunk_concurrent_limit, mut codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) });
        };
        codec_options.set_store_empty_chunks(write_empty_chunks);

        let p = self.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let _keep_alive = value_ref;
            let p = &p;
            let input = match input {
                InputValue::Array(ptr, len) => {
                    // SAFETY: `value_ref` keeps the array alive until this future
                    // completes. As in the sync pipeline, the caller must not
                    // mutate it while the write is in flight.
                    // FIXME: Handle variable length data types, convert value to bytes and offsets
                    let input_slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
                    Ok(ArrayBytes::new_flen(Cow::Borrowed(input_slice)))
                }
                InputValue::Constant(constant_value) => Err(constant_value),
            };
            let input = &input;
            let codec_options = &codec_options;
            stream::iter(chunk_descriptions.into_iter().map(Ok::<ChunkItem, PyErr>))
                .try_for_each_concurrent(chunk_concurrent_limit, move |item| async move {
                    let chunk_subset_bytes = match input {
                        Ok(input) => input
                            .extract_array_subset(
                                &item.subset,
                                bytemuck::must_cast_slice(&item.array_shape),
                                &p.data_type,
                            )
                            .map_codec_err()?,
                        Err(constant_value) => ArrayBytes::new_fill_value(
                            &p.data_type,
                            item.chunk_subset.num_elements(),
                            constant_value,
                        )
                        .map_py_err::<PyRuntimeError>()?,
                    };
                    p.store_chunk_subset_bytes(&item, chunk_subset_bytes, codec_options)
                        .await
                })
                .await
        })
    }
}
