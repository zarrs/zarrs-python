use crate::utils::{cartesian_product, update_bytes_flen, update_bytes_flen_with_indexer};
use dlpark::prelude::*;
use numpy::ndarray::{Array, ArrayBase, ArrayViewD};
use numpy::{PyArray, PyArray2, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyList, PySlice, PyTuple};
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use rayon::prelude::*;
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use std::ffi::c_void;
use std::fmt::Display;
use std::ops::Range;
use zarrs::array::codec::CodecOptionsBuilder;
use zarrs::array::{Array as RustArray, ArrayCodecTraits, RecommendedConcurrency, UnsafeCellSlice};
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

#[pyclass]
pub struct ZarrsPythonArray {
    pub arr: RustArray<dyn ReadableStorageTraits + 'static>,
}

impl ZarrsPythonArray {
    fn maybe_convert_u64(&self, ind: i32, axis: usize) -> PyResult<u64> {
        let mut ind_u64: u64 = ind as u64;
        if ind < 0 {
            if self.arr.shape()[axis] as i32 + ind < 0 {
                return Err(PyIndexError::new_err(format!("{0} out of bounds", ind)));
            }
            ind_u64 =
                u64::try_from(ind).map_err(|_| PyIndexError::new_err("Failed to extract start"))?;
        }
        return Ok(ind_u64);
    }

    fn bound_slice(&self, slice: &Bound<PySlice>, axis: usize) -> PyResult<Range<u64>> {
        let start: i32 = slice.getattr("start")?.extract().map_or(0, |x| x);
        let stop: i32 = slice
            .getattr("stop")?
            .extract()
            .map_or(self.arr.shape()[axis] as i32, |x| x);
        let start_u64 = self.maybe_convert_u64(start, 0)?;
        let stop_u64 = self.maybe_convert_u64(stop, 0)?;
        // let _step: u64 = slice.getattr("step")?.extract().map_or(1, |x| x); // there is no way to use step it seems with zarrs?
        let selection = start_u64..stop_u64;
        return Ok(selection);
    }

    pub fn fill_from_slices(&self, slices: Vec<Range<u64>>) -> PyResult<Vec<Range<u64>>> {
        Ok(self
            .arr
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
            .collect())
    }

    fn extract_coords(
        &self,
        chunk_coords_and_selections: &Bound<'_, PyList>,
    ) -> PyResult<Vec<Vec<u64>>> {
        chunk_coords_and_selections
            .into_iter()
            .map(|chunk_coord_and_selection| {
                if let Ok(chunk_coord_and_selection_tuple) =
                    chunk_coord_and_selection.downcast::<PyTuple>()
                {
                    let coord = chunk_coord_and_selection_tuple.get_item(0)?;
                    let coord_extracted: Vec<u64>;
                    if let Ok(coord_downcast) = coord.downcast::<PyTuple>() {
                        coord_extracted = coord_downcast.extract()?;
                        return Ok(coord_extracted);
                    } else if let Ok(nd_array) = coord.downcast::<PyArray2<u64>>() {
                        let nd_array_extracted: Vec<u64> = nd_array.to_vec()?;
                        return Ok(nd_array_extracted);
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "Cannot take {0}, must be int, ndarray, or slice",
                            coord.to_string()
                        )));
                    }
                }
                return Err(PyTypeError::new_err(format!(
                    "Unsupported type: {0}",
                    chunk_coord_and_selection
                )));
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
                if let Ok(chunk_coord_and_selection_tuple) =
                    chunk_coord_and_selection.downcast::<PyTuple>()
                {
                    let selection = chunk_coord_and_selection_tuple.get_item(index)?;
                    if let Ok(slice) = selection.downcast::<PySlice>() {
                        return Ok(ArraySubset::new_with_ranges(
                            &self.fill_from_slices(vec![self.bound_slice(slice, 0)?])?,
                        ));
                    } else if let Ok(tuple) = selection.downcast::<PyTuple>() {
                        let ranges: Vec<Range<u64>> = tuple
                            .into_iter()
                            .enumerate()
                            .map(|(index, val)| {
                                if let Ok(int) = val.downcast::<PyInt>() {
                                    let end = self.maybe_convert_u64(int.extract()?, index)?;
                                    Ok(end..(end + 1))
                                } else if let Ok(slice) = val.downcast::<PySlice>() {
                                    Ok(self.bound_slice(slice, index)?)
                                } else {
                                    return Err(PyValueError::new_err(format!(
                                        "Cannot take {0}, must be int or slice",
                                        val.to_string()
                                    )));
                                }
                            })
                            .collect::<Result<Vec<Range<u64>>, _>>()?;
                        return Ok(ArraySubset::new_with_ranges(
                            &self.fill_from_slices(ranges)?,
                        ));
                    } else {
                        return Err(PyTypeError::new_err(format!(
                            "Unsupported type: {0}",
                            selection
                        )));
                    }
                }
                return Err(PyTypeError::new_err(format!(
                    "Unsupported type: {0}",
                    chunk_coord_and_selection
                )));
            })
            .collect::<PyResult<Vec<ArraySubset>>>()
    }

    fn extract_selection_to_vec_indices(
        &self,
        chunk_coords_and_selections: &Bound<'_, PyList>,
        index: usize,
    ) -> PyResult<Vec<Vec<Vec<i64>>>> {
        chunk_coords_and_selections
            .into_iter()
            .map(|chunk_coord_and_selection| {
                if let Ok(chunk_coord_and_selection_tuple) =
                    chunk_coord_and_selection.downcast::<PyTuple>()
                {
                    let selection = chunk_coord_and_selection_tuple.get_item(index)?;
                    if let Ok(tuple) = selection.downcast::<PyTuple>() {
                        let res = tuple
                            .into_iter()
                            .map(|(val)| {
                                if let Ok(nd_array) = val.downcast::<PyArrayDyn<i64>>() {
                                    let res = nd_array.to_vec()?;
                                    Ok(res)
                                } else {
                                    Err(PyTypeError::new_err(format!(
                                        "Unsupported type: {0}",
                                        tuple
                                    )))
                                }
                            })
                            .collect::<PyResult<Vec<Vec<i64>>>>()?;
                        return Ok(res);
                    } else {
                        return Err(PyTypeError::new_err(format!(
                            "Unsupported type: {0}",
                            selection
                        )));
                    }
                }
                return Err(PyTypeError::new_err(format!(
                    "Unsupported type: {0}",
                    chunk_coord_and_selection
                )));
            })
            .collect::<PyResult<Vec<Vec<Vec<i64>>>>>()
    }

    fn is_selection_numpy_array(
        &self,
        chunk_coords_and_selections: &Bound<'_, PyList>,
        index: usize,
    ) -> bool {
        let results = chunk_coords_and_selections
            .into_iter()
            .map(|chunk_coord_and_selection| {
                if let Ok(chunk_coord_and_selection_tuple) =
                    chunk_coord_and_selection.downcast::<PyTuple>()
                {
                    let selection = chunk_coord_and_selection_tuple.get_item(index);
                    if let Ok(selection_unwrapped) = selection {
                        if let Ok(tuple) = selection_unwrapped.downcast::<PyTuple>() {
                            let res: Vec<bool> = tuple
                                .into_iter()
                                .map(|(val)| -> bool {
                                    let nd_array = val.downcast::<PyArrayDyn<i64>>();
                                    let res = match nd_array {
                                        Ok(_) => true,
                                        Err(_) => false,
                                    };
                                    return res;
                                })
                                .collect();
                            return res;
                        }
                        return vec![false];
                    }
                    return vec![false];
                }
                return vec![false];
            })
            .flatten()
            .collect::<Vec<bool>>();
        results.iter().any(|x: &bool| *x)
    }
}

#[pymethods]
impl ZarrsPythonArray {
    pub fn retrieve_chunk_subset(
        &self,
        out_shape: &Bound<'_, PyTuple>,
        chunk_coords_and_selections: &Bound<'_, PyList>,
    ) -> PyResult<ManagerCtx<PyZarrArr>> {
        if let Ok(chunk_coords_and_selection_list) =
            chunk_coords_and_selections.downcast::<PyList>()
        {
            // Need to scale up everything because zarr's chunks don't match zarrs' chunks
            let chunk_representation = self
                .arr
                .chunk_array_representation(&vec![0; self.arr.chunk_grid().dimensionality()])
                .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?;
            let data_type_size = chunk_representation.data_type().size();
            let out_shape_extracted = out_shape
                .into_iter()
                .map(|x| x.extract::<u64>())
                .collect::<PyResult<Vec<u64>>>()?;
            let coords_extracted = &self.extract_coords(chunk_coords_and_selection_list)?;
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
            let codec_options = CodecOptionsBuilder::new()
                .concurrent_target(codec_concurrent_target)
                .build();
            let size_output = out_shape_extracted.iter().product::<u64>() as usize;
            let mut output = Vec::with_capacity(size_output * data_type_size);

            if self.is_selection_numpy_array(chunk_coords_and_selections, 1) {
                let selections_extracted =
                    self.extract_selection_to_vec_indices(chunk_coords_and_selections, 1)?;
                let borrowed_selections = &selections_extracted;
                {
                    let output = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut output);
                    let retrieve_chunk = |chunk: NdArrayChunk| {
                        let indices: Vec<u64> = cartesian_product(chunk.selection)
                            .iter()
                            .map(|x| {
                                x.iter().enumerate().fold(0, |acc, (ind, x)| {
                                    acc + (*x as u64)
                                        * if (ind + 1 == chunk.selection.len()) {
                                            1
                                        } else {
                                            self.arr.chunk_shape(&chunk.index).unwrap()[(ind + 1)..]
                                                .iter()
                                                .map(|x| x.get() as u64)
                                                .product::<u64>()
                                        }
                                })
                            })
                            .collect();
                        let chunk_subset_bytes = self
                            .arr
                            .retrieve_chunk(&chunk.index)
                            .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?;
                        update_bytes_flen_with_indexer(
                            unsafe { output.get() },
                            &out_shape_extracted,
                            &chunk_subset_bytes,
                            &chunk.out_selection,
                            &indices,
                            data_type_size,
                        );
                        Ok::<_, PyErr>(())
                    };
                    let zipped_iterator = coords_extracted
                        .into_iter()
                        .zip(borrowed_selections.into_iter())
                        .zip(out_selections_extracted.into_iter())
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
                return Ok(ManagerCtx::new(PyZarrArr {
                    shape: out_shape_extracted,
                    arr: output,
                    dtype: chunk_representation.data_type().clone(),
                }));
            }
            let selections_extracted =
                self.extract_selection_to_array_subset(chunk_coords_and_selections, 1)?;
            let out_selections_extracted =
                &self.extract_selection_to_array_subset(chunk_coords_and_selections, 2)?;
            let borrowed_selections = &selections_extracted;
            {
                let output = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut output);
                let retrieve_chunk = |chunk: Chunk| {
                    let chunk_subset_bytes = self
                        .arr
                        .retrieve_chunk_subset_opt(&chunk.index, &chunk.selection, &codec_options)
                        .map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()))?;
                    update_bytes_flen(
                        unsafe { output.get() },
                        &out_shape_extracted,
                        &chunk_subset_bytes,
                        &chunk.out_selection,
                        data_type_size,
                    );
                    Ok::<_, PyErr>(())
                };
                let zipped_iterator = coords_extracted
                    .into_iter()
                    .zip(borrowed_selections.into_iter())
                    .zip(out_selections_extracted.into_iter())
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
                dtype: chunk_representation.data_type().clone(),
            }))
        } else {
            return Err(PyTypeError::new_err(format!(
                "Unsupported type: {0}",
                chunk_coords_and_selections
            )));
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
