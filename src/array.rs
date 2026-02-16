use std::borrow::Cow;

use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use zarrs::array::concurrency::concurrency_chunks_and_codec;
use zarrs::array::{
    Array, ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArrayCodecTraits,
    ArrayIndicesTinyVec, ArraySubset, CodecOptions,
};
use zarrs::plugin::{ExtensionName, ZarrVersion};
use zarrs::storage::ReadableWritableListableStorage;

use crate::store::StoreConfig;
use crate::utils::{PyErrExt as _, nparray_to_slice, nparray_to_unsafe_cell_slice};

fn data_type_to_numpy_str(dt: &zarrs::array::DataType) -> PyResult<String> {
    let name = dt
        .name(ZarrVersion::V3)
        .ok_or_else(|| PyErr::new::<PyTypeError, _>("unknown data type"))?;
    match name.as_ref() {
        "bool" | "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32"
        | "uint64" | "float16" | "float32" | "float64" | "complex64" | "complex128" => {
            Ok(name.into_owned())
        }
        _ => Err(PyErr::new::<PyTypeError, _>(format!(
            "unsupported data type for numpy: {name}"
        ))),
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct ArrayImpl {
    array: Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
    codec_options: CodecOptions,
    data_type_size: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl ArrayImpl {
    #[pyo3(signature = (
        store_config,
        path,
        *,
        validate_checksums=false,
        chunk_concurrent_minimum=None,
        num_threads=None,
        direct_io=false,
    ))]
    #[new]
    fn new(
        mut store_config: StoreConfig,
        path: &str,
        validate_checksums: bool,
        chunk_concurrent_minimum: Option<usize>,
        num_threads: Option<usize>,
        direct_io: bool,
    ) -> PyResult<Self> {
        store_config.direct_io(direct_io);
        let store: ReadableWritableListableStorage =
            (&store_config).try_into().map_py_err::<PyTypeError>()?;
        let array = Array::open(store, path).map_py_err::<PyRuntimeError>()?;
        let data_type_size = array.data_type().fixed_size().ok_or_else(|| {
            PyErr::new::<PyTypeError, _>("variable length data type not supported")
        })?;
        let mut codec_options = CodecOptions::default()
            .with_validate_checksums(validate_checksums)
            .with_concurrent_target(num_threads.unwrap_or(rayon::current_num_threads()));
        if let Some(chunk_concurrent_minimum) = chunk_concurrent_minimum {
            codec_options.set_chunk_concurrent_minimum(chunk_concurrent_minimum);
        }
        Ok(Self {
            array,
            codec_options,
            data_type_size,
        })
    }

    #[getter]
    fn shape(&self) -> Vec<u64> {
        self.array.shape().to_vec()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.array.dimensionality()
    }

    #[getter]
    fn dtype(&self) -> PyResult<String> {
        data_type_to_numpy_str(self.array.data_type())
    }

    fn retrieve(
        &self,
        py: Python,
        ranges: Vec<(u64, u64)>,
        output: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        let subset_ranges: Vec<std::ops::Range<u64>> =
            ranges.iter().map(|&(start, stop)| start..stop).collect();
        let array_subset = ArraySubset::new_with_ranges(&subset_ranges);

        let output_cell_slice = nparray_to_unsafe_cell_slice(output)?;
        let output_shape: Vec<u64> = output
            .shape()
            .iter()
            .map(|&s| u64::try_from(s).map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{e}"))))
            .collect::<PyResult<Vec<u64>>>()?;

        let data_type_size = self.data_type_size;
        let full_output_subset = ArraySubset::new_with_shape(output_shape.clone());

        py.detach(move || {
            let mut output_view = unsafe {
                // SAFETY: we are the sole writer to this output array
                ArrayBytesFixedDisjointView::new(
                    output_cell_slice,
                    data_type_size,
                    &output_shape,
                    full_output_subset,
                )
                .map_py_err::<PyRuntimeError>()?
            };
            let target = ArrayBytesDecodeIntoTarget::Fixed(&mut output_view);
            self.array
                .retrieve_array_subset_into_opt(&array_subset, target, &self.codec_options)
                .map_py_err::<PyRuntimeError>()
        })
    }

    fn store(
        &self,
        py: Python,
        ranges: Vec<(u64, u64)>,
        input: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        let subset_ranges: Vec<std::ops::Range<u64>> =
            ranges.iter().map(|&(start, stop)| start..stop).collect();
        let array_subset = ArraySubset::new_with_ranges(&subset_ranges);

        let input_slice = nparray_to_slice(input)?;
        let array_bytes = ArrayBytes::new_flen(Cow::Borrowed(input_slice));

        py.detach(move || {
            self.array
                .store_array_subset_opt(&array_subset, array_bytes, &self.codec_options)
                .map_py_err::<PyRuntimeError>()
        })
    }

    fn copy_from(
        &self,
        py: Python,
        source: &ArrayImpl,
        source_ranges: Vec<(u64, u64)>,
        dest_ranges: Vec<(u64, u64)>,
    ) -> PyResult<()> {
        let source_subset = ArraySubset::new_with_ranges(
            &source_ranges.iter().map(|&(s, e)| s..e).collect::<Vec<_>>(),
        );
        let dest_subset = ArraySubset::new_with_ranges(
            &dest_ranges.iter().map(|&(s, e)| s..e).collect::<Vec<_>>(),
        );

        if source_subset.shape() != dest_subset.shape() {
            return Err(PyErr::new::<PyValueError, _>(
                "source and destination region shapes must match",
            ));
        }

        py.detach(move || {
            let chunks = self
                .array
                .chunks_in_array_subset(&dest_subset)
                .map_py_err::<PyRuntimeError>()?
                .ok_or_else(|| {
                    PyErr::new::<PyRuntimeError, _>("failed to compute overlapping chunks")
                })?;
            let num_chunks = chunks.num_elements_usize();

            // Calculate chunk/codec concurrency
            let chunk_shape = self
                .array
                .chunk_shape(&vec![0; self.array.dimensionality()])
                .map_py_err::<PyRuntimeError>()?;
            let codec_concurrency = self
                .array
                .codecs()
                .recommended_concurrency(&chunk_shape, self.array.data_type())
                .map_py_err::<PyRuntimeError>()?;
            let (chunk_concurrent_limit, codec_options) = concurrency_chunks_and_codec(
                self.codec_options.concurrent_target(),
                num_chunks,
                &self.codec_options,
                &codec_concurrency,
            );

            let dest_start = dest_subset.start();
            let source_start = source_subset.start();
            // let source_cache = ArrayShardedReadableExtCache::new(&source.array);

            let copy_chunk = |chunk_indices: ArrayIndicesTinyVec| -> PyResult<()> {
                let chunk_subset = self
                    .array
                    .chunk_subset_bounded(&chunk_indices)
                    .map_py_err::<PyRuntimeError>()?;

                let overlap = chunk_subset
                    .overlap(&dest_subset)
                    .map_py_err::<PyRuntimeError>()?;
                if overlap.is_empty() {
                    return Ok(());
                }

                // Map overlap coordinates from dest space to source space
                let source_overlap_ranges: Vec<std::ops::Range<u64>> = overlap
                    .to_ranges()
                    .iter()
                    .enumerate()
                    .map(|(i, range)| {
                        let offset = range.start - dest_start[i] + source_start[i];
                        let len = range.end - range.start;
                        offset..(offset + len)
                    })
                    .collect();
                let source_overlap = ArraySubset::new_with_ranges(&source_overlap_ranges);

                // NOTE: Could retrieve into a pre-allocated buffer (per thread) with `regular` grid
                let data: ArrayBytes = source
                    .array
                    .retrieve_array_subset_opt(&source_overlap, &source.codec_options)
                    // .retrieve_array_subset_sharded_opt(
                    //     &source_cache,
                    //     &source_overlap,
                    //     &source.codec_options,
                    // )
                    .map_py_err::<PyRuntimeError>()?;

                self.array
                    .store_array_subset_opt(&overlap, data, &codec_options)
                    .map_py_err::<PyRuntimeError>()?;

                Ok(())
            };

            let indices = chunks.indices();
            iter_concurrent_limit!(chunk_concurrent_limit, indices, try_for_each, copy_chunk)?;

            Ok(())
        })
    }
}
