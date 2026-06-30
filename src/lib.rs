#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::borrow::Cow;
use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use zarrs::array::{ArrayMetadata, CodecChain, CodecOptions, DataType, FillValue};
use zarrs::config::global_config;
use zarrs::convert::array_metadata_v2_to_v3;
use zarrs::plugin::ZarrVersion;

mod async_pipeline;
mod chunk_item;
mod concurrency;
mod runtime;
mod store;
mod sync_pipeline;
#[cfg(test)]
mod tests;
mod utils;

use crate::async_pipeline::AsyncCodecPipelineImpl;
use crate::sync_pipeline::CodecPipelineImpl;
use crate::utils::PyErrExt as _;

/// Configuration parsed from the array metadata, shared by the synchronous and
/// asynchronous codec pipelines (everything except the store, which differs).
pub(crate) struct ArrayConfig {
    pub(crate) codec_chain: Arc<CodecChain>,
    pub(crate) codec_options: CodecOptions,
    pub(crate) chunk_concurrent_minimum: usize,
    pub(crate) chunk_concurrent_maximum: usize,
    pub(crate) num_threads: usize,
    pub(crate) fill_value: FillValue,
    pub(crate) data_type: DataType,
}

pub(crate) fn parse_array_config(
    array_metadata: &str,
    validate_checksums: bool,
    chunk_concurrent_minimum: Option<usize>,
    chunk_concurrent_maximum: Option<usize>,
    num_threads: Option<usize>,
) -> PyResult<ArrayConfig> {
    let metadata = serde_json::from_str(array_metadata).map_py_err::<PyTypeError>()?;
    let metadata_v3 = match &metadata {
        ArrayMetadata::V2(v2) => {
            Cow::Owned(array_metadata_v2_to_v3(v2).map_py_err::<PyTypeError>()?)
        }
        ArrayMetadata::V3(v3) => Cow::Borrowed(v3),
    };
    let codec_chain =
        Arc::new(CodecChain::from_metadata(&metadata_v3.codecs).map_py_err::<PyTypeError>()?);
    let codec_options = CodecOptions::default().with_validate_checksums(validate_checksums);

    let chunk_concurrent_minimum =
        chunk_concurrent_minimum.unwrap_or(global_config().chunk_concurrent_minimum());
    let chunk_concurrent_maximum = chunk_concurrent_maximum.unwrap_or(rayon::current_num_threads());
    let num_threads = num_threads.unwrap_or(rayon::current_num_threads());

    let data_type = DataType::from_metadata(&metadata_v3.data_type).map_py_err::<PyTypeError>()?;
    let fill_value = data_type
        .fill_value(&metadata_v3.fill_value, ZarrVersion::V3)
        .or_else(|_| {
            Err(match &metadata {
                ArrayMetadata::V2(metadata) => format!(
                    "incompatible fill value metadata: dtype={}, fill_value={}",
                    metadata.dtype, metadata.fill_value
                ),
                ArrayMetadata::V3(metadata) => format!(
                    "incompatible fill value metadata: data_type={}, fill_value={}",
                    metadata.data_type, metadata.fill_value
                ),
            })
        })
        .map_py_err::<PyTypeError>()?;

    Ok(ArrayConfig {
        codec_chain,
        codec_options,
        chunk_concurrent_minimum,
        chunk_concurrent_maximum,
        num_threads,
        fill_value,
        data_type,
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<CodecPipelineImpl>()?;
    m.add_class::<AsyncCodecPipelineImpl>()?;
    m.add_class::<chunk_item::ChunkItem>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
