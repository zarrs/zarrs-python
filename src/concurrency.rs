use pyo3::{exceptions::PyRuntimeError, PyErr, PyResult};
use zarrs::array::{
    codec::CodecOptions, concurrency::calc_concurrency_outer_inner, ArrayCodecTraits,
    RecommendedConcurrency,
};

use crate::{chunk_item::ChunksItem, CodecPipelineImpl};

pub trait ChunkConcurrentLimitAndCodecOptions {
    fn get_chunk_concurrent_limit_and_codec_options(
        &self,
        codec_pipeline_impl: &CodecPipelineImpl,
    ) -> PyResult<Option<(usize, CodecOptions)>>;
}

impl<T> ChunkConcurrentLimitAndCodecOptions for Vec<T>
where
    T: ChunksItem,
{
    fn get_chunk_concurrent_limit_and_codec_options(
        &self,
        codec_pipeline_impl: &CodecPipelineImpl,
    ) -> PyResult<Option<(usize, CodecOptions)>> {
        let num_chunks = self.len();
        let Some(chunk_descriptions0) = self.first() else {
            return Ok(None);
        };
        let chunk_representation = chunk_descriptions0.representation();

        let codec_concurrency = codec_pipeline_impl
            .codec_chain
            .recommended_concurrency(chunk_representation)
            .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?;

        let min_concurrent_chunks =
            std::cmp::min(codec_pipeline_impl.chunk_concurrent_minimum, num_chunks);
        let max_concurrent_chunks =
            std::cmp::max(codec_pipeline_impl.chunk_concurrent_maximum, num_chunks);
        let (chunk_concurrent_limit, codec_concurrent_limit) = calc_concurrency_outer_inner(
            codec_pipeline_impl.num_threads,
            &RecommendedConcurrency::new(min_concurrent_chunks..max_concurrent_chunks),
            &codec_concurrency,
        );
        let codec_options = codec_pipeline_impl
            .codec_options
            .into_builder()
            .concurrent_target(codec_concurrent_limit)
            .build();
        Ok(Some((chunk_concurrent_limit, codec_options)))
    }
}
