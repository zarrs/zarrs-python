use std::fmt::Display;

use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::{Bound, PyErr, PyResult, PyTypeInfo};

pub(crate) trait PyErrExt<T> {
    fn map_py_err<PE: PyTypeInfo>(self) -> PyResult<T>;
}

impl<T, E: Display> PyErrExt<T> for Result<T, E> {
    fn map_py_err<PE: PyTypeInfo>(self) -> PyResult<T> {
        self.map_err(|e| PyErr::new::<PE, _>(format!("{e}")))
    }
}

pub(crate) trait PyUntypedArrayExt {
    fn shape_zarr(&self) -> PyResult<Vec<u64>>;
}

impl PyUntypedArrayExt for Bound<'_, PyUntypedArray> {
    fn shape_zarr(&self) -> PyResult<Vec<u64>> {
        Ok(if self.shape().is_empty() {
            vec![1] // scalar value
        } else {
            self.shape()
                .iter()
                .map(|&i| u64::try_from(i))
                .collect::<Result<_, _>>()?
        })
    }
}
