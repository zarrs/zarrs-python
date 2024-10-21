use numpy::{IntoPyArray, PyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use zarrs::array::codec::CodecOptionsBuilder;
use zarrs::array::{Array as RustArray, ArrayCodecTraits, RecommendedConcurrency};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::ReadableStorageTraits;

#[allow(clippy::module_name_repetitions)]
#[pyclass]
pub struct ZarrsPythonArray {
    pub arr: RustArray<dyn ReadableStorageTraits + 'static>,
    pub n_threads: usize,
}

#[pymethods]
impl ZarrsPythonArray {
    pub fn retrieve_chunks<'py>(
        &self,
        py: Python<'py>,
        chunk_coords: Vec<Vec<u64>>,
    ) -> PyResult<Vec<Bound<'py, PyArray<u8, numpy::ndarray::Dim<[usize; 1]>>>>> {
        let chunks = ArraySubset::new_with_shape(self.arr.chunk_grid_shape().unwrap());
        let chunk_representation = self
            .arr
            .chunk_array_representation(&vec![0; self.arr.chunk_grid().dimensionality()])
            .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?;
        let concurrent_target = std::thread::available_parallelism().unwrap().get();
        let (chunks_concurrent_limit, codec_concurrent_target) =
            zarrs::array::concurrency::calc_concurrency_outer_inner(
                concurrent_target,
                &{
                    let concurrent_chunks =
                        std::cmp::min(chunks.num_elements_usize(), self.n_threads);
                    RecommendedConcurrency::new_minimum(concurrent_chunks)
                },
                &self
                    .arr
                    .codecs()
                    .recommended_concurrency(&chunk_representation)
                    .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?,
            );
        let codec_options = CodecOptionsBuilder::new()
            .concurrent_target(codec_concurrent_target)
            .build();
        let retrieve_chunk = |chunk_index: &Vec<u64>| -> Result<Vec<u8>, PyErr> {
            Ok(self
                .arr
                .retrieve_chunk_opt(chunk_index, &codec_options)
                .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?
                .into_fixed()
                .expect("zarrs-python does not support variable-sized data types")
                .into_owned())
        };
        iter_concurrent_limit!(chunks_concurrent_limit, &chunk_coords, map, retrieve_chunk)
            .map(|x| x.unwrap())
            .collect::<Vec<Vec<u8>>>()
            .into_iter()
            .map(|bytes| {
                let pyarray = bytes.into_pyarray_bound(py);
                Ok(pyarray)
            })
            .collect::<PyResult<Vec<Bound<'py, PyArray<u8, numpy::ndarray::Dim<[usize; 1]>>>>>>()
    }
}
