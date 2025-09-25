use std::fmt::Display;

use pyo3::{PyErr, PyResult, PyTypeInfo};
use zarrs::array::codec::CodecError;

pub(crate) trait PyErrStrExt<T> {
    fn map_py_err_from_str<PE: PyTypeInfo>(self) -> PyResult<T>;
}

impl<T, E: Display> PyErrStrExt<T> for Result<T, E> {
    fn map_py_err_from_str<PE: PyTypeInfo>(self) -> PyResult<T> {
        self.map_err(|e| PyErr::new::<PE, _>(format!("{e}")))
    }
}

pub(crate) trait PyErrExt<T> {
    fn map_py_err(self) -> PyResult<T>;
}

impl<T> PyErrExt<T> for Result<T, CodecError> {
    fn map_py_err(self) -> PyResult<T> {
        // see https://docs.python.org/3/library/exceptions.html#exception-hierarchy
        self.map_err(|e| match e {
            // requested indexing operation doesn’t match shape
            CodecError::IncompatibleIndexer(_)
            | CodecError::IncompatibleDimensionalityError(_)
            | CodecError::InvalidByteRangeError(_) => {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("{e}"))
            }
            // some pipe, file, or subprocess failed
            CodecError::IOError(_) => PyErr::new::<pyo3::exceptions::PyOSError, _>(format!("{e}")),
            // all the rest: some unknown runtime problem
            e => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")),
        })
    }
}
