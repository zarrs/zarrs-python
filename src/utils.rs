use std::fmt::Display;

use pyo3::{
    Borrowed, Bound, FromPyObject, IntoPyObject, PyAny, PyErr, PyResult, PyTypeInfo, Python,
    exceptions::PyValueError, types::PyString,
};
use zarrs::array::{CodecError, codec::SubchunkWriteOrder};

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

pub fn is_whole_chunk(item: &ChunkItem) -> bool {
    item.chunk_subset.start().iter().all(|&o| o == 0)
        && item.chunk_subset.shape() == bytemuck::must_cast_slice::<_, u64>(&item.shape)
}

#[derive(Debug, Clone, Copy)]
pub struct SubchunkWriteOrderWrapper(pub SubchunkWriteOrder);

// zarr-python's sharding codec exposes `subchunk_write_order` as one of
// morton / unordered / lexicographic / colexicographic. zarrs can only write
// C (== lexicographic) and Unordered, so the orders it can't reproduce fall
// back to Unordered — data is correct, only the physical byte layout differs.
// ponytail: map morton/colexicographic -> Unordered until zarrs can write them.
impl<'py> IntoPyObject<'py> for SubchunkWriteOrderWrapper {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            SubchunkWriteOrder::C => Ok("lexicographic".into_pyobject(py)?),
            SubchunkWriteOrder::Unordered | SubchunkWriteOrder::Random => {
                Ok("unordered".into_pyobject(py)?)
            }
            _ => Err(PyValueError::new_err(
                "Unrecognized subchunk write order for converting to python object.",
            )),
        }
    }
}

impl<'py> FromPyObject<'_, 'py> for SubchunkWriteOrderWrapper {
    type Error = PyErr;

    fn extract(option: Borrowed<'_, 'py, PyAny>) -> PyResult<SubchunkWriteOrderWrapper> {
        let order = match option.extract::<&str>()? {
            "lexicographic" => SubchunkWriteOrder::C,
            "unordered" | "morton" | "colexicographic" => SubchunkWriteOrder::Unordered,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unrecognized subchunk write order {other:?}, only `morton`, `unordered`, `lexicographic` and `colexicographic` allowed.",
                )));
            }
        };
        Ok(SubchunkWriteOrderWrapper(order))
    }
}

impl pyo3_stub_gen::PyStubType for SubchunkWriteOrderWrapper {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module(
            "typing.Literal['morton', 'unordered', 'lexicographic', 'colexicographic']",
            "typing".into(),
        )
    }
}
