use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use zarrs::array::{Array as RustArray};
use zarrs::storage::ReadableStorageTraits;
use pyo3::types::PySlice;

#[pyclass]
pub struct Array {
    pub arr: RustArray<dyn ReadableStorageTraits + 'static>
}

#[pymethods]
impl Array {
    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        if let Ok(slice) = key.downcast::<PySlice>() {
            let start: i32 = slice.getattr("start")?.extract().map_or(0, |x| x);
            let mut start_u64: u64 = start as u64;
            if start < 0 {
                if self.arr.shape()[0] as i32 + start < 0 {
                    return Err(PyIndexError::new_err(format!("{0} out of bounds", start)))
                }
                start_u64 = u64::try_from(start).map_err(|_| PyErr::new::<PyIndexError, _>("Failed to extract start"))?;
            }
            let stop: i32 = slice.getattr("stop")?.extract().map_or(self.arr.shape()[0] as i32, |x| x);
            let mut stop_u64: u64 = stop as u64;
            if stop < 0 {
                if self.arr.shape()[0] as i32 + stop < 0 {
                    return Err(PyIndexError::new_err(format!("{0} out of bounds", stop)))
                }
                stop_u64 = u64::try_from(stop).map_err(|_| PyErr::new::<PyIndexError, _>("Failed to extract stop"))?;
            }
            let _step: u64 = slice.getattr("step")?.extract().map_or(1, |x| x);
            let selection: Vec<u64> = (start_u64..stop_u64).step_by(_step.try_into().unwrap()).collect();
            println!("{:?}", selection);
            return self.arr.retrieve_chunk(&selection[..]).map_err(|x| PyErr::new::<PyTypeError, _>(x.to_string()));
        } else {
            return Err(PyTypeError::new_err("Unsupported type"));
        }
    }
}
