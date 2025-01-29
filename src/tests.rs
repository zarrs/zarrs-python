use pyo3::ffi::c_str;

use numpy::PyUntypedArray;
use pyo3::{
    types::{PyAnyMethods, PyModule},
    Bound, PyResult, Python,
};

use crate::CodecPipelineImpl;

#[test]
fn test_nparray_to_unsafe_cell_slice_empty() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let arr: Bound<'_, PyUntypedArray> = PyModule::from_code(
            py,
            c_str!(
                "def empty_array():
                import numpy as np
                return np.empty(0, dtype=np.uint8)"
            ),
            c_str!(""),
            c_str!(""),
        )?
        .getattr("empty_array")?
        .call0()?
        .extract()?;

        let slice = CodecPipelineImpl::nparray_to_unsafe_cell_slice(&arr)?;
        assert!(slice.is_empty());
        Ok(())
    })
}
