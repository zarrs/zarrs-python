use crate::utils::{cartesian_product, update_bytes_flen, update_bytes_flen_with_indexer};
use dlpark::prelude::*;
use numpy::{PyArray2, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyList, PySlice, PyTuple};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use std::ffi::c_void;
use std::fmt::Display;
use std::ops::Range;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::CodecOptionsBuilder;
use zarrs::array::{Array as RustArray, ArrayCodecTraits, RecommendedConcurrency};
use zarrs::array_subset::ArraySubset;
use zarrs::config::global_config;
use zarrs::storage::ReadableStorageTraits;

#[derive(Debug)]
struct Chunk<'a> {
    index: &'a Vec<u64>,
    selection: &'a ArraySubset,
    out_selection: &'a ArraySubset,
}

impl Display for Chunk<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{index len: {:?}, selection: {:?}, out_selection: {:?}}}",
            self.index.len(),
            self.selection,
            self.out_selection
        )
    }
}

#[derive(Debug)]
struct NdArrayChunk<'a> {
    index: &'a Vec<u64>,
    selection: &'a Vec<Vec<i64>>,
    out_selection: &'a ArraySubset,
}

impl Display for NdArrayChunk<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{index len: {:?}, selection len: {:?}, out_selection: {:?}}}",
            self.index.len(),
            self.selection.len(),
            self.out_selection
        )
    }
}

#[allow(clippy::module_name_repetitions)]
#[pyclass]
pub struct ZarrsPythonArray {
    pub arr: RustArray<dyn ReadableStorageTraits + 'static>,
}

// First some extraction utilities for going from python to rust
impl ZarrsPythonArray {
    fn maybe_convert_u64(&self, ind: i64, axis: usize) -> PyResult<u64> {
        if ind >= 0 {
            return Ok(ind.try_into()?);
        }
        match self.arr.shape()[axis].checked_add_signed(ind) {
            Some(x) => Ok(x),
            None => Err(PyIndexError::new_err(format!("{ind} out of bounds"))),
        }
    }

    fn bound_slice(&self, slice: &Bound<PySlice>, axis: usize) -> PyResult<Range<u64>> {
        let start: u64 = slice
            .getattr("start")?
            .extract::<i64>()
            .map_or(Ok(0), |x| self.maybe_convert_u64(x, 0))?;
        let stop: u64 = slice.getattr("stop")?.extract().map_or_else(
            |_| Ok(self.arr.shape()[axis]),
            |x| self.maybe_convert_u64(x, 0),
        )?;
        // let _step: u64 = slice.getattr("step")?.extract().map_or(1, |x| x); // there is no way to use step it seems with zarrs?
        Ok(start..stop)
    }

    pub fn fill_from_slices(&self, slices: &[Range<u64>]) -> std::vec::Vec<std::ops::Range<u64>> {
        self.arr
            .shape()
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                if index < slices.len() {
                    slices[index].clone()
                } else {
                    0..value
                }
            })
            .collect()
    }

    fn extract_coords(chunk_coords_and_selections: &Bound<'_, PyList>) -> PyResult<Vec<Vec<u64>>> {
        chunk_coords_and_selections
            .into_iter()
            .map(|chunk_coord_and_selection| {
                let Ok(chunk_coord_and_selection_tuple) =
                    chunk_coord_and_selection.downcast::<PyTuple>()
                else {
                    return Err(PyTypeError::new_err(format!(
                        "Unsupported type: {chunk_coord_and_selection}"
                    )));
                };
                let coord = chunk_coord_and_selection_tuple.get_item(0)?;
                let coord_extracted: Vec<u64>;
                if let Ok(coord_downcast) = coord.downcast::<PyTuple>() {
                    coord_extracted = coord_downcast.extract()?;
                    return Ok(coord_extracted);
                }
                if let Ok(nd_array) = coord.downcast::<PyArray2<u64>>() {
                    let nd_array_extracted: Vec<u64> = nd_array.to_vec()?;
                    return Ok(nd_array_extracted);
                }
                Err(PyValueError::new_err(format!(
                    "Cannot take {coord}, must be int, ndarray, or slice"
                )))
            })
            .collect::<PyResult<Vec<Vec<u64>>>>()
    }

    fn extract_selection_to_array_subset(
        &self,
        chunk_coords_and_selections: &Bound<'_, PyList>,
        index: usize,
    ) -> PyResult<Vec<ArraySubset>> {
        chunk_coords_and_selections
            .into_iter()
            .map(|chunk_coord_and_selection| {
                let Ok(chunk_coord_and_selection_tuple) =
                    chunk_coord_and_selection.downcast::<PyTuple>()
                else {
                    return Err(PyTypeError::new_err(format!(
                        "Unsupported type: {chunk_coord_and_selection}"
                    )));
                };
                let selection = chunk_coord_and_selection_tuple.get_item(index)?;
                if let Ok(slice) = selection.downcast::<PySlice>() {
                    return Ok(ArraySubset::new_with_ranges(
                        &self.fill_from_slices(&[self.bound_slice(slice, 0)?]),
                    ));
                }
                let Ok(tuple) = selection.downcast::<PyTuple>() else {
                    return Err(PyTypeError::new_err(format!(
                        "Unsupported type: {selection}"
                    )));
                };
                if tuple.len() == 0 {
                    return Ok(ArraySubset::new_with_ranges(
                        #[allow(clippy::single_range_in_vec_init)]
                        &self.fill_from_slices(&[0..1]),
                    ));
                }
                let ranges: Vec<Range<u64>> = tuple
                    .into_iter()
                    .enumerate()
                    .map(|(axis, val)| {
                        if let Ok(int) = val.downcast::<PyInt>() {
                            let end = self.maybe_convert_u64(int.extract()?, axis)?;
                            #[allow(clippy::range_plus_one)]
                            Ok(end..end + 1)
                        } else if let Ok(slice) = val.downcast::<PySlice>() {
                            self.bound_slice(slice, axis)
                        } else {
                            Err(PyValueError::new_err(format!(
                                "Cannot take {val}, must be int or slice"
                            )))
                        }
                    })
                    .collect::<Result<Vec<Range<u64>>, _>>()?;
                Ok(ArraySubset::new_with_ranges(
                    &self.fill_from_slices(&ranges),
                ))
            })
            .collect::<PyResult<Vec<ArraySubset>>>()
    }

    fn extract_selection_to_vec_indices(
        chunk_coords_and_selections: &Bound<'_, PyList>,
        index: usize,
    ) -> PyResult<Vec<Vec<Vec<i64>>>> {
        chunk_coords_and_selections
            .into_iter()
            .map(|chunk_coord_and_selection| {
                let Ok(chunk_coord_and_selection_tuple) =
                    chunk_coord_and_selection.downcast::<PyTuple>()
                else {
                    return Err(PyTypeError::new_err(format!(
                        "Unsupported type: {chunk_coord_and_selection}"
                    )));
                };
                let selection = chunk_coord_and_selection_tuple.get_item(index)?;
                let Ok(tuple) = selection.downcast::<PyTuple>() else {
                    return Err(PyTypeError::new_err(format!(
                        "Unsupported type: {selection}"
                    )));
                };
                let res = tuple
                    .into_iter()
                    .map(|val| {
                        if let Ok(nd_array) = val.downcast::<PyArrayDyn<i64>>() {
                            let res = nd_array.to_vec()?;
                            Ok(res)
                        } else {
                            Err(PyTypeError::new_err(format!("Unsupported type: {tuple}")))
                        }
                    })
                    .collect::<PyResult<Vec<Vec<i64>>>>()?;
                Ok(res)
            })
            .collect::<PyResult<Vec<Vec<Vec<i64>>>>>()
    }

    fn is_selection_numpy_array(
        chunk_coords_and_selections: &Bound<'_, PyList>,
        index: usize,
    ) -> bool {
        let results = chunk_coords_and_selections
            .into_iter()
            .flat_map(|chunk_coord_and_selection| {
                let Some(selection_unwrapped) = chunk_coord_and_selection
                    .downcast::<PyTuple>()
                    .ok()
                    .and_then(|sel| sel.get_item(index).ok())
                else {
                    return vec![false];
                };
                let Ok(tuple) = selection_unwrapped.downcast::<PyTuple>() else {
                    return vec![false];
                };
                tuple
                    .into_iter()
                    .map(|val| -> bool { val.downcast::<PyArrayDyn<i64>>().is_ok() })
                    .collect::<Vec<bool>>()
            })
            .collect::<Vec<bool>>();
        results.iter().any(|x: &bool| *x)
    }
}

impl ZarrsPythonArray {
    #[allow(clippy::too_many_arguments)]
    fn get_data_from_primitive_selection(
        &self,
        chunk_coords_and_selections: &pyo3::Bound<'_, PyList>,
        out_shape_extracted: Vec<u64>,
        data_type_size: usize,
        coords_extracted: &[Vec<u64>],
        out_selections_extracted: &[ArraySubset],
        chunks_concurrent_limit: usize,
        codec_concurrent_target: usize,
        size_output: usize,
        dtype: zarrs::array::DataType,
    ) -> PyResult<ManagerCtx<PyZarrArr>> {
        let mut output = Vec::with_capacity(size_output * data_type_size);
        let codec_options = CodecOptionsBuilder::new()
            .concurrent_target(codec_concurrent_target)
            .build();
        let selections_extracted =
            self.extract_selection_to_array_subset(chunk_coords_and_selections, 1)?;
        let borrowed_selections = &selections_extracted;
        {
            let output = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut output);
            let retrieve_chunk = |chunk: Chunk| {
                let output = unsafe { output.as_mut_slice() };
                let chunk_subset_bytes = self
                    .arr
                    .retrieve_chunk_subset_opt(chunk.index, chunk.selection, &codec_options)
                    .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?
                    .into_fixed()
                    .expect("zarrs-python does not support variable-sized data types");
                update_bytes_flen(
                    output,
                    &out_shape_extracted,
                    &chunk_subset_bytes,
                    chunk.out_selection,
                    data_type_size,
                );
                Ok::<_, PyErr>(())
            };
            let zipped_iterator = coords_extracted
                .iter()
                .zip(borrowed_selections)
                .zip(out_selections_extracted)
                .map(|((index, selection), out_selection)| Chunk {
                    index,
                    selection,
                    out_selection,
                });
            iter_concurrent_limit!(
                chunks_concurrent_limit,
                zipped_iterator.collect::<Vec<Chunk>>(),
                try_for_each,
                retrieve_chunk
            )?;
        }
        unsafe { output.set_len(size_output) };
        Ok(ManagerCtx::new(PyZarrArr {
            shape: out_shape_extracted,
            arr: output,
            dtype,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn get_data_from_numpy_selection(
        &self,
        chunk_coords_and_selections: &pyo3::Bound<'_, PyList>,
        out_shape_extracted: Vec<u64>,
        data_type_size: usize,
        coords_extracted: &[Vec<u64>],
        out_selections_extracted: &Vec<ArraySubset>,
        chunks_concurrent_limit: usize,
        codec_concurrent_target: usize,
        size_output: usize,
        dtype: zarrs::array::DataType,
    ) -> PyResult<ManagerCtx<PyZarrArr>> {
        let mut output = Vec::with_capacity(size_output * data_type_size);
        let selections_extracted: Vec<Vec<Vec<i64>>> =
            ZarrsPythonArray::extract_selection_to_vec_indices(chunk_coords_and_selections, 1)?;
        let codec_options = CodecOptionsBuilder::new()
            .concurrent_target(codec_concurrent_target)
            .build();
        let borrowed_selections = &selections_extracted;
        {
            let output = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut output);
            let retrieve_chunk = |chunk: NdArrayChunk| {
                let chunk_shape = self.arr.chunk_shape(chunk.index).unwrap();
                let indices: Vec<_> = cartesian_product(chunk.selection)
                    .into_iter()
                    .map(|x| {
                        x.into_iter().enumerate().fold(0, |acc: u64, (ind, x)| {
                            let factor = if ind + 1 == chunk.selection.len() {
                                1
                            } else {
                                chunk_shape[(ind + 1)..]
                                    .iter()
                                    .map(|x| x.get())
                                    .product::<u64>()
                                    .try_into()
                                    .unwrap()
                            };
                            acc.saturating_add_signed(x * factor)
                        })
                    })
                    .collect();
                let chunk_subset_bytes = self
                    .arr
                    .retrieve_chunk_opt(chunk.index, &codec_options)
                    .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?
                    .into_fixed()
                    .expect("zarrs-python does not support variable-sized data types");
                update_bytes_flen_with_indexer(
                    unsafe { output.as_mut_slice() },
                    &out_shape_extracted,
                    &chunk_subset_bytes,
                    chunk.out_selection,
                    &indices,
                    data_type_size,
                );
                Ok::<_, PyErr>(())
            };
            let zipped_iterator = coords_extracted
                .iter()
                .zip(borrowed_selections)
                .zip(out_selections_extracted)
                .map(|((index, selection), out_selection)| NdArrayChunk {
                    index,
                    selection,
                    out_selection,
                });
            iter_concurrent_limit!(
                chunks_concurrent_limit,
                zipped_iterator.collect::<Vec<NdArrayChunk>>(),
                try_for_each,
                retrieve_chunk
            )?;
        }
        unsafe { output.set_len(size_output) };
        Ok(ManagerCtx::new(PyZarrArr {
            shape: out_shape_extracted,
            arr: output,
            dtype,
        }))
    }
}

#[pymethods]
impl ZarrsPythonArray {
    pub fn retrieve_chunk_subset(
        &self,
        out_shape: &Bound<'_, PyTuple>,
        chunk_coords_and_selections: &Bound<'_, PyList>,
    ) -> PyResult<ManagerCtx<PyZarrArr>> {
        let Ok(chunk_coords_and_selection_list) = chunk_coords_and_selections.downcast::<PyList>()
        else {
            return Err(PyTypeError::new_err(format!(
                "Unsupported type: {chunk_coords_and_selections}"
            )));
        };

        // Need to scale up everything because zarr's chunks don't match zarrs' chunks
        let chunk_representation = self
            .arr
            .chunk_array_representation(&vec![0; self.arr.chunk_grid().dimensionality()])
            .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?;
        let data_type_size = chunk_representation
            .data_type()
            .fixed_size()
            .expect("zarrs-python does not support variable-sized data types");
        let out_shape_extracted = out_shape
            .into_iter()
            .map(|x| x.extract::<u64>())
            .collect::<PyResult<Vec<_>>>()?;
        let coords_extracted = ZarrsPythonArray::extract_coords(chunk_coords_and_selection_list)?;
        let out_selections_extracted =
            &self.extract_selection_to_array_subset(chunk_coords_and_selections, 2)?;
        let chunks = ArraySubset::new_with_shape(self.arr.chunk_grid_shape().unwrap());
        let concurrent_target = std::thread::available_parallelism().unwrap().get();
        let (chunks_concurrent_limit, codec_concurrent_target) =
            zarrs::array::concurrency::calc_concurrency_outer_inner(
                concurrent_target,
                &{
                    let concurrent_chunks = std::cmp::min(
                        chunks.num_elements_usize(),
                        global_config().chunk_concurrent_minimum(),
                    );
                    RecommendedConcurrency::new_minimum(concurrent_chunks)
                },
                &self
                    .arr
                    .codecs()
                    .recommended_concurrency(&chunk_representation)
                    .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?,
            );
        let size_output = out_shape_extracted.iter().product::<u64>().try_into()?;
        let dtype = chunk_representation.data_type().clone();
        if ZarrsPythonArray::is_selection_numpy_array(chunk_coords_and_selections, 1) {
            self.get_data_from_numpy_selection(
                chunk_coords_and_selections,
                out_shape_extracted,
                data_type_size,
                &coords_extracted,
                out_selections_extracted,
                chunks_concurrent_limit,
                codec_concurrent_target,
                size_output,
                dtype,
            )
        } else {
            self.get_data_from_primitive_selection(
                chunk_coords_and_selections,
                out_shape_extracted,
                data_type_size,
                &coords_extracted,
                out_selections_extracted,
                chunks_concurrent_limit,
                codec_concurrent_target,
                size_output,
                dtype,
            )
        }
    }
}

pub struct PyZarrArr {
    arr: Vec<u8>,
    shape: Vec<u64>,
    dtype: zarrs::array::DataType,
}

impl ToTensor for PyZarrArr {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.arr.as_ptr().cast::<c_void>().cast_mut()
    }

    fn shape_and_strides(&self) -> ShapeAndStrides {
        ShapeAndStrides::new_contiguous_with_strides(
            self.shape
                .iter()
                .map(|x| {
                    (*x).try_into()
                        .expect("Array is too big to be converted into a tensor")
                })
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
