//! Asynchronous codec pipeline.
//!
//! [`AsyncCodecPipelineImpl`] mirrors [`crate::CodecPipelineImpl`], but holds an
//! [`AsyncReadableWritableListableStorage`] and talks to it with `async`
//! `get`/`set`/`erase` calls instead of going through the
//! `AsyncToSyncStorageAdapter`.
//!
//! [`tokio::task::spawn`] / [`JoinSet`](tokio::task::JoinSet) require their
//! futures to be `Send + 'static`, so a task that decoded *directly into* that
//! buffer would not compile: the view borrows `value` and cannot satisfy
//! `'static` (and we would also have to argue disjoint-write soundness across
//! threads we do not control).
//!
//! So retrieval is split into two phases:
//!
//! 1. **Fetch (parallel concurrent-fetch/decode, `tokio`)
//! 2. **Fill (synchronous, parallel).**
//!    [`ArrayBytesFixedDisjointView::copy_from_slice`] over its disjoint subset.
//!    This touches the non-`'static` borrow, but it is plain synchronous code —
//!    no `tokio`, no `'static` requirement.
//!
//! The trade-off is that all decoded chunk subsets are held in memory at once
//! (≈ the size of `value`) before the fill begins.
//!
//! The write path has no such conflict — it only *reads* the input buffer — so it
//! still drives `async` stores concurrently inside a single
//! [`crate::runtime::block_on`] via [`futures`] combinators.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use futures::stream::{self, TryStreamExt};
use itertools::Itertools;
use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::task::JoinSet;
use zarrs::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayToBytesCodecTraits,
    AsyncArrayPartialDecoderTraits, CodecChain, CodecOptions, DataType, FillValue,
    update_array_bytes,
};
use zarrs::storage::{
    AsyncReadableStorageTraits, AsyncReadableWritableListableStorage, AsyncWritableStorageTraits,
    StorageHandle, StoreKey,
};
// Not re-exported by `zarrs` (only the async traits are); see Cargo.toml.
use zarrs_codec::AsyncStoragePartialDecoder;

use crate::chunk_item::ChunkItem;
use crate::concurrency::{ChunkConcurrentLimitAndCodecOptions, CodecPipelineConfig};
use crate::runtime::block_on;
use crate::store::StoreConfig;
use crate::utils::{PyCodecErrExt, PyErrExt as _, is_whole_chunk};
use crate::{ArrayConfig, CodecPipelineImpl, parse_array_config};

struct RetrieveContext {
    store: AsyncReadableWritableListableStorage,
    codec_chain: Arc<CodecChain>,
    codec_options: CodecOptions,
    data_type: DataType,
    fill_value: FillValue,
}

#[gen_stub_pyclass]
#[pyclass]
pub struct AsyncCodecPipelineImpl {
    pub(crate) store: AsyncReadableWritableListableStorage,
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
            .store
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

    async fn store_chunk_bytes(
        &self,
        item: &ChunkItem,
        value_decoded: ArrayBytes<'_>,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        value_decoded
            .validate(item.num_elements, &self.data_type)
            .map_codec_err()?;

        if value_decoded.is_fill_value(&self.fill_value) {
            self.store
                .erase(&item.key)
                .await
                .map_py_err::<PyRuntimeError>()
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
            self.store
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
    ) -> PyResult<Self> {
        // `direct_io` only affects the (synchronous) filesystem store, which the
        // async pipeline does not support; kept for signature parity.
        store_config.direct_io(direct_io);
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

        Ok(Self {
            store,
            codec_chain,
            codec_options,
            chunk_concurrent_minimum,
            chunk_concurrent_maximum,
            num_threads,
            fill_value,
            data_type,
        })
    }

    fn retrieve_chunks_and_apply_index(
        &self,
        py: Python,
        chunk_descriptions: Vec<ChunkItem>, // FIXME: Ref / iterable?
        value: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        // `output` is an `UnsafeCellSlice` borrowing `value`'s buffer; its
        // lifetime is *not* `'static`. It is never handed to a `tokio` task —
        // only the synchronous fill phase below touches it (see module docs).
        let output = CodecPipelineImpl::nparray_to_unsafe_cell_slice(value)?;

        // Only the codec options are needed here; chunk-level parallelism is
        // delegated to the tokio runtime rather than a manual concurrency limit.
        let Some((_chunk_concurrent_limit, codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return Ok(());
        };

        // FIXME: the fill phase only supports fixed length data types. For
        // variable length data types, need a codepath without `copy_from_slice`.
        let data_type_size = self
            .data_type
            .fixed_size()
            .ok_or("variable length data type not supported")
            .map_py_err::<PyTypeError>()?;

        // Unique partial (non-whole) chunks each need an async partial decoder,
        // so that subsets sharing a shard reuse a single decoder.
        let partial_chunk_items: Vec<ChunkItem> = chunk_descriptions
            .iter()
            .filter(|item| !is_whole_chunk(item))
            .unique_by(|item| item.key.clone())
            .cloned()
            .collect();

        // Owned, `'static` config cloned into each fetch task.
        let ctx = Arc::new(RetrieveContext {
            store: self.store.clone(),
            codec_chain: self.codec_chain.clone(),
            codec_options,
            data_type: self.data_type.clone(),
            fill_value: self.fill_value.clone(),
        });

        py.detach(move || {
            // Phase 1: fetch + decode every chunk's subset into an owned
            // `ArrayBytes`, in parallel across the tokio runtime's worker threads.
            let decoded: Vec<ArrayBytes<'static>> = block_on(async {
                // Build the partial decoders (one per unique shard) in parallel.
                let mut cache_tasks: JoinSet<
                    PyResult<(StoreKey, Arc<dyn AsyncArrayPartialDecoderTraits>)>,
                > = JoinSet::new();
                for item in partial_chunk_items {
                    let ctx = ctx.clone();
                    cache_tasks.spawn(async move {
                        let storage_handle = Arc::new(StorageHandle::new(ctx.store.clone()));
                        let input_handle = Arc::new(AsyncStoragePartialDecoder::new(
                            storage_handle,
                            item.key.clone(),
                        ));
                        let partial_decoder = ctx
                            .codec_chain
                            .clone()
                            .async_partial_decoder(
                                input_handle,
                                &item.shape,
                                &ctx.data_type,
                                &ctx.fill_value,
                                &ctx.codec_options,
                            )
                            .await
                            .map_codec_err()?;
                        Ok((item.key, partial_decoder))
                    });
                }
                let mut partial_decoder_cache: HashMap<
                    StoreKey,
                    Arc<dyn AsyncArrayPartialDecoderTraits>,
                > = HashMap::new();
                while let Some(joined) = cache_tasks.join_next().await {
                    let (key, decoder) = joined.map_py_err::<PyRuntimeError>()??;
                    partial_decoder_cache.insert(key, decoder);
                }

                // One fetch task per chunk. Each returns owned bytes, so the task
                // is `Send + 'static` and runs with real tokio parallelism.
                let mut fetch_tasks: JoinSet<PyResult<(usize, ArrayBytes<'static>)>> =
                    JoinSet::new();
                for (index, item) in chunk_descriptions.iter().enumerate() {
                    let ctx = ctx.clone();
                    let item = item.clone();
                    let partial_decoder = partial_decoder_cache.get(&item.key).cloned();
                    fetch_tasks.spawn(async move {
                        let bytes: ArrayBytes<'static> = if is_whole_chunk(&item) {
                            // See zarrs::array::Array::async_retrieve_chunk_opt
                            match ctx
                                .store
                                .get(&item.key)
                                .await
                                .map_py_err::<PyRuntimeError>()?
                            {
                                Some(chunk_encoded) => {
                                    let chunk_encoded: Vec<u8> = chunk_encoded.into();
                                    ctx.codec_chain
                                        .decode(
                                            chunk_encoded.into(),
                                            &item.shape,
                                            &ctx.data_type,
                                            &ctx.fill_value,
                                            &ctx.codec_options,
                                        )
                                        .map_codec_err()?
                                        .into_owned()
                                }
                                // Missing chunk: the subset is the fill value.
                                None => ArrayBytes::new_fill_value(
                                    &ctx.data_type,
                                    item.num_elements,
                                    &ctx.fill_value,
                                )
                                .map_py_err::<PyRuntimeError>()?
                                .into_owned(),
                            }
                        } else {
                            // See zarrs::array::Array::async_retrieve_chunk_subset_opt
                            let partial_decoder = partial_decoder.ok_or_else(|| {
                                PyRuntimeError::new_err(format!(
                                    "Partial decoder not found for key: {}",
                                    item.key
                                ))
                            })?;
                            // `into_owned` here drops the borrow of the task-local
                            // partial decoder so the bytes can leave the task.
                            partial_decoder
                                .partial_decode(&item.chunk_subset, &ctx.codec_options)
                                .await
                                .map_codec_err()?
                                .into_owned()
                        };
                        Ok((index, bytes))
                    });
                }

                let mut decoded: Vec<Option<ArrayBytes<'static>>> =
                    (0..chunk_descriptions.len()).map(|_| None).collect();
                while let Some(joined) = fetch_tasks.join_next().await {
                    let (index, bytes) = joined.map_py_err::<PyRuntimeError>()??;
                    decoded[index] = Some(bytes);
                }
                // Every index is produced exactly once by the loop above.
                Ok::<_, PyErr>(
                    decoded
                        .into_iter()
                        .map(|bytes| bytes.expect("every chunk index is decoded exactly once"))
                        .collect(),
                )
            })?;

            // Phase 2: copy each decoded subset into `value`. Synchronous, so the
            // non-`'static` borrow of the output buffer is sound.
            for (item, bytes) in chunk_descriptions.iter().zip(decoded) {
                let mut output_view = unsafe {
                    // SAFETY: chunks represent disjoint array subsets, so the
                    // views never write overlapping output bytes.
                    ArrayBytesFixedDisjointView::new(
                        output,
                        data_type_size,
                        bytemuck::must_cast_slice(&item.array_shape),
                        item.subset.clone(),
                    )
                    .map_py_err::<PyRuntimeError>()?
                };
                let bytes = bytes.into_fixed().map_py_err::<PyRuntimeError>()?;
                output_view
                    .copy_from_slice(&bytes)
                    .map_py_err::<PyRuntimeError>()?;
            }
            Ok(())
        })
    }

    fn store_chunks_with_indices(
        &self,
        py: Python,
        chunk_descriptions: Vec<ChunkItem>,
        value: &Bound<'_, PyUntypedArray>,
        write_empty_chunks: bool,
    ) -> PyResult<()> {
        enum InputValue<'a> {
            Array(ArrayBytes<'a>),
            Constant(FillValue),
        }

        // Get input array
        let input_slice = CodecPipelineImpl::nparray_to_slice(value)?;
        let input = if value.ndim() > 0 {
            // FIXME: Handle variable length data types, convert value to bytes and offsets
            InputValue::Array(ArrayBytes::new_flen(Cow::Borrowed(input_slice)))
        } else {
            InputValue::Constant(FillValue::new(input_slice.to_vec()))
        };

        // Adjust the concurrency based on the codec chain and the first chunk description
        let Some((chunk_concurrent_limit, mut codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return Ok(());
        };
        codec_options.set_store_empty_chunks(write_empty_chunks);

        py.detach(move || {
            block_on(async move {
                let input = &input;
                let codec_options = &codec_options;
                stream::iter(chunk_descriptions.into_iter().map(Ok::<ChunkItem, PyErr>))
                    .try_for_each_concurrent(chunk_concurrent_limit, move |item| async move {
                        let chunk_subset_bytes = match input {
                            InputValue::Array(input) => input
                                .extract_array_subset(
                                    &item.subset,
                                    bytemuck::must_cast_slice(&item.array_shape),
                                    &self.data_type,
                                )
                                .map_codec_err()?,
                            InputValue::Constant(constant_value) => ArrayBytes::new_fill_value(
                                &self.data_type,
                                item.chunk_subset.num_elements(),
                                constant_value,
                            )
                            .map_py_err::<PyRuntimeError>()?,
                        };
                        self.store_chunk_subset_bytes(&item, chunk_subset_bytes, codec_options)
                            .await
                    })
                    .await
            })
        })
    }
}
