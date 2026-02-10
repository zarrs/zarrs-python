use std::fmt::Display;
use std::ptr::NonNull;

use numpy::npyffi::PyArrayObject;
use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::{Bound, PyErr, PyResult, PyTypeInfo};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::CodecError;

use crate::ChunkItem;

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

pub fn is_whole_chunk(item: &ChunkItem) -> bool {
    item.chunk_subset.start().iter().all(|&o| o == 0)
        && item.chunk_subset.shape() == bytemuck::must_cast_slice::<_, u64>(&item.shape)
}

pub fn py_untyped_array_to_array_object<'a>(
    value: &'a Bound<'_, PyUntypedArray>,
) -> &'a PyArrayObject {
    // TODO: Upstream a PyUntypedArray.as_array_ref()?
    //       https://github.com/zarrs/zarrs-python/pull/80/files/75be39184905d688ac04a5f8bca08c5241c458cd#r1918365296
    let array_object_ptr: NonNull<PyArrayObject> = NonNull::new(value.as_array_ptr())
        .expect("bug in numpy crate: Bound<'_, PyUntypedArray>::as_array_ptr unexpectedly returned a null pointer");
    let array_object: &'a PyArrayObject = unsafe {
        // SAFETY: the array object pointed to by array_object_ptr is valid for 'a
        array_object_ptr.as_ref()
    };
    array_object
}

pub fn nparray_to_slice<'a>(value: &'a Bound<'_, PyUntypedArray>) -> Result<&'a [u8], PyErr> {
    if !value.is_c_contiguous() {
        return Err(PyErr::new::<PyValueError, _>(
            "input array must be a C contiguous array".to_string(),
        ));
    }
    let array_object: &PyArrayObject = py_untyped_array_to_array_object(value);
    let array_data = array_object.data.cast::<u8>();
    let array_len = value.len() * value.dtype().itemsize();
    let slice = unsafe {
        // SAFETY: array_data is a valid pointer to a u8 array of length array_len
        debug_assert!(!array_data.is_null());
        std::slice::from_raw_parts(array_data, array_len)
    };
    Ok(slice)
}

pub fn nparray_to_unsafe_cell_slice<'a>(
    value: &'a Bound<'_, PyUntypedArray>,
) -> Result<UnsafeCellSlice<'a, u8>, PyErr> {
    if !value.is_c_contiguous() {
        return Err(PyErr::new::<PyValueError, _>(
            "input array must be a C contiguous array".to_string(),
        ));
    }
    let array_object: &PyArrayObject = py_untyped_array_to_array_object(value);
    let array_data = array_object.data.cast::<u8>();
    let array_len = value.len() * value.dtype().itemsize();
    let output = unsafe {
        // SAFETY: array_data is a valid pointer to a u8 array of length array_len
        debug_assert!(!array_data.is_null());
        std::slice::from_raw_parts_mut(array_data, array_len)
    };
    Ok(UnsafeCellSlice::new(output))
}
