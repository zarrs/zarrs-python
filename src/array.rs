use dlpark::prelude::*;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use std::ffi::c_void;
use zarrs::array::codec::CodecOptionsBuilder;
use zarrs::array::{Array as RustArray, ArrayCodecTraits, RecommendedConcurrency};
use zarrs::array_subset::ArraySubset;
use zarrs::config::global_config;
use zarrs::storage::ReadableStorageTraits;

#[allow(clippy::module_name_repetitions)]
#[pyclass]
pub struct ZarrsPythonArray {
    pub arr: RustArray<dyn ReadableStorageTraits + 'static>,
    pub n_threads: usize,
}

#[pymethods]
impl ZarrsPythonArray {
    pub fn retrieve_chunks(
        &self,
        chunk_coords: Vec<Vec<u64>>,
    ) -> PyResult<Vec<ManagerCtx<PyZarrArr>>> {
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
        let dtype = chunk_representation.data_type().clone();
        let retrieve_chunk = |chunk_index: &Vec<u64>| -> Result<std::borrow::Cow<[u8]>, PyErr> {
            Ok(self
                .arr
                .retrieve_chunk_opt(chunk_index, &codec_options)
                .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?
                .into_fixed()
                .expect("zarrs-python does not support variable-sized data types"))
        };
        let bytes_vec =
            iter_concurrent_limit!(chunks_concurrent_limit, &chunk_coords, map, retrieve_chunk)
                .map(|chunk_bytes| -> Vec<u8> { chunk_bytes.unwrap().into_owned() })
                .collect::<Vec<Vec<u8>>>();
        bytes_vec
            .into_iter()
            .zip(&chunk_coords)
            .map(|(bytes, chunk_index)| {
                let chunk_size: Vec<u64> = self
                    .arr
                    .chunk_shape(chunk_index)
                    .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?
                    .to_array_shape();
                Ok(ManagerCtx::new(PyZarrArr {
                    arr: bytes,
                    shape: chunk_size,
                    dtype: dtype.clone(),
                }))
            })
            .collect::<PyResult<Vec<ManagerCtx<PyZarrArr>>>>()
    }
}

pub struct PyZarrArr {
    arr: Vec<u8>,
    shape: Vec<u64>,
    dtype: zarrs::array::DataType,
}

impl ToTensor for PyZarrArr {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.arr.as_ptr() as *const c_void as *mut c_void
    }
    fn shape_and_strides(&self) -> ShapeAndStrides {
        ShapeAndStrides::new_contiguous_with_strides(
            self.shape
                .iter()
                .map(|x| *x as i64)
                .collect::<Vec<i64>>()
                .iter(),
        )
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn dtype(&self) -> DataType {
        match self.dtype {
            zarrs::array::DataType::Int16 => DataType::I16,
            zarrs::array::DataType::Int32 => DataType::I32,
            zarrs::array::DataType::Int64 => DataType::I64,
            zarrs::array::DataType::Int8 => DataType::I8,
            zarrs::array::DataType::UInt16 => DataType::U16,
            zarrs::array::DataType::UInt32 => DataType::U32,
            zarrs::array::DataType::UInt64 => DataType::U64,
            zarrs::array::DataType::UInt8 => DataType::U8,
            zarrs::array::DataType::Float32 => DataType::F32,
            zarrs::array::DataType::Float64 => DataType::F64,
            zarrs::array::DataType::Bool => DataType::BOOL,
            _ => panic!("Unsupported data type"),
        }
    }
}
