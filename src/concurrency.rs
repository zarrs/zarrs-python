use pyo3::PyResult;
use zarrs::array::{
    ArrayCodecTraits, CodecChain, CodecOptions, DataType, RecommendedConcurrency,
    concurrency::calc_concurrency_outer_inner,
};

use crate::{chunk_item::ChunkItem, utils::PyCodecErrExt as _};

/// The pieces of pipeline configuration needed to compute concurrency limits.
///
/// Implemented by both the synchronous [`crate::CodecPipelineImpl`] and the
/// asynchronous [`crate::async_pipeline::AsyncCodecPipelineImpl`], so the
/// concurrency calculation can be shared between them.
pub trait CodecPipelineConfig {
    fn codec_chain(&self) -> &CodecChain;
    fn data_type(&self) -> &DataType;
    fn codec_options(&self) -> &CodecOptions;
    fn chunk_concurrent_minimum(&self) -> usize;
    fn chunk_concurrent_maximum(&self) -> usize;
    fn num_threads(&self) -> usize;
}

pub trait ChunkConcurrentLimitAndCodecOptions {
    fn get_chunk_concurrent_limit_and_codec_options<C: CodecPipelineConfig>(
        &self,
        config: &C,
    ) -> PyResult<Option<(usize, CodecOptions)>>;
}

impl ChunkConcurrentLimitAndCodecOptions for Vec<ChunkItem> {
    fn get_chunk_concurrent_limit_and_codec_options<C: CodecPipelineConfig>(
        &self,
        config: &C,
    ) -> PyResult<Option<(usize, CodecOptions)>> {
        let num_chunks = self.len();
        let Some(item) = self.first() else {
            return Ok(None);
        };

        let codec_concurrency = config
            .codec_chain()
            .recommended_concurrency(&item.shape, config.data_type())
            .map_codec_err()?;

        let min_concurrent_chunks = std::cmp::min(config.chunk_concurrent_minimum(), num_chunks);
        let max_concurrent_chunks = std::cmp::max(config.chunk_concurrent_maximum(), num_chunks);
        let (chunk_concurrent_limit, codec_concurrent_limit) = calc_concurrency_outer_inner(
            config.num_threads(),
            &RecommendedConcurrency::new(min_concurrent_chunks..max_concurrent_chunks),
            &codec_concurrency,
        );
        let codec_options = config
            .codec_options()
            .with_concurrent_target(codec_concurrent_limit);
        Ok(Some((chunk_concurrent_limit, codec_options)))
    }
}
