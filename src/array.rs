use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use zarrs::array::{Array as RustArray};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::ReadableStorageTraits;
use pyo3::types::{PyInt, PyList, PySlice};
use std::ops::Range;

#[pyclass]
pub struct Array {
    pub arr: RustArray<dyn ReadableStorageTraits + 'static>
}

impl Array {

    fn maybe_convert_u64(&self, ind: i32, axis: usize) -> PyResult<u64> {
        let mut ind_u64: u64 = ind as u64;
        if ind < 0 {
            if self.arr.shape()[axis] as i32 + ind < 0 {
                return Err(PyIndexError::new_err(format!("{0} out of bounds", ind)))
            }
            ind_u64 = u64::try_from(ind).map_err(|_| PyIndexError::new_err("Failed to extract start"))?;
        }
        return Ok(ind_u64);
    }

    fn bound_slice(&self, slice: &Bound<PySlice>, axis: usize) -> PyResult<Range<u64>> {
        let start: i32 = slice.getattr("start")?.extract().map_or(0, |x| x);
        let stop: i32 = slice.getattr("stop")?.extract().map_or(self.arr.shape()[axis] as i32, |x| x);
        let start_u64 = self.maybe_convert_u64(start, 0)?;
        let stop_u64 = self.maybe_convert_u64(stop, 0)?;
        // let _step: u64 = slice.getattr("step")?.extract().map_or(1, |x| x); // there is no way to use step it seems with zarrs?
        let selection = start_u64..stop_u64;
        return Ok(selection)
    }

    fn fill_from_slices(&self, slices: Vec<Range<u64>>) -> PyResult<Vec<Range<u64>>> {
        Ok(self.arr.shape().iter().enumerate().map(|(index, &value)| { if index < slices.len() { slices[index].clone() } else { 0..value } }).collect())
    }
}

#[pymethods]
impl Array {

    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        let selection: ArraySubset;
        if let Ok(slice) = key.downcast::<PySlice>() {
            selection = ArraySubset::new_with_ranges(&self.fill_from_slices(vec![self.bound_slice(slice, 0)?])?);
        } else if let Ok(list) = key.downcast::<PyList>(){
            let ranges: Vec<Range<u64>> = list.into_iter().enumerate().map(|(index, val)| {
                if let Ok(int) = val.downcast::<PyInt>() {
                    let end = self.maybe_convert_u64(int.extract()?, index)?;
                    Ok(end..(end + 1))
                } else if let Ok(slice) = val.downcast::<PySlice>() {
                    Ok(self.bound_slice(slice, index)?)
                } else {
                    return Err(PyValueError::new_err(format!("Cannot take {0}, must be int or slice", val.to_string())));
                }
            }).collect::<Result<Vec<Range<u64>>, _>>()?;
            selection = ArraySubset::new_with_ranges(&self.fill_from_slices(ranges)?);
        } else {
            return Err(PyTypeError::new_err("Unsupported type"));
        }
        return self.arr.retrieve_array_subset(&selection).map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()));
    }
}


