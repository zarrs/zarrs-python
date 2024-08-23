use pyo3::{exceptions::PyTypeError, prelude::*};
use std::sync::Arc;
use zarrs::array::Array as RustArray;
use zarrs::storage::{store, ReadableStorage};

mod array;
mod utils;

#[pyfunction]
fn open_array(path: &str) -> PyResult<array::ZarrsPythonArray> {
    let s: ReadableStorage;
    if path.starts_with("http://") | path.starts_with("https://") {
        s = Arc::new(store::HTTPStore::new(path).or_else(|x| utils::err(x.to_string()))?);
    } else {
        s = Arc::new(store::FilesystemStore::new(path).or_else(|x| utils::err(x.to_string()))?);
    }
    let arr = RustArray::new(s, &"/").or_else(|x| utils::err(x.to_string()))?;
    Ok(array::ZarrsPythonArray { arr })
}

/// A Python module implemented in Rust.
#[pymodule]
fn zarrs_python_internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open_array, m)?)?;
    m.add_class::<array::ZarrsPythonArray>()?;
    Ok(())
}
