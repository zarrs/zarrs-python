use std::fmt::Display;

use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::{Bound, PyErr, PyResult, PyTypeInfo};
use zarrs::array::codec::CodecError;

use crate::{ChunksItem, WithSubset};

pub(crate) trait PyErrExt<T> {
    fn map_py_err<PE: PyTypeInfo>(self) -> PyResult<T>;
}

impl<T, E: Display> PyErrExt<T> for Result<T, E> {
    fn map_py_err<PE: PyTypeInfo>(self) -> PyResult<T> {
        self.map_err(|e| PyErr::new::<PE, _>(format!("{e}")))
    }
}

pub(crate) trait PyCodecErrExt<T> {
    fn map_codec_err(self) -> PyResult<T>;
}

impl<T> PyCodecErrExt<T> for Result<T, CodecError> {
    fn map_codec_err(self) -> PyResult<T> {
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

pub fn is_whole_chunk(item: &WithSubset) -> bool {
    item.chunk_subset.start().iter().all(|&o| o == 0)
        && item.chunk_subset.shape() == item.representation().shape_u64()
}
