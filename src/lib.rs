use pyo3::{exceptions::PyTypeError, prelude::*};
use std::sync::Arc;
use zarrs::storage::{ReadableStorage, store};
use zarrs::array::Array as RustArray;

mod array;
mod utils;

#[pyfunction]
fn open_array(path: &str) -> PyResult<array::Array> {
    const HTTP_URL: &str =
        "https://raw.githubusercontent.com/LDeakin/zarrs/main/tests/data/array_write_read.zarr";
    const ARRAY_PATH: &str = "/group/array";

    let s: ReadableStorage = Arc::new(store::HTTPStore::new(HTTP_URL).or_else(|x| utils::err(x.to_string()))?);
    let arr  = RustArray::new(s, ARRAY_PATH).or_else(|x| utils::err(x.to_string()))?; 
    Ok(array::Array{ arr })
}

/// A Python module implemented in Rust.
#[pymodule]
fn zarrs_python(_py: Python, m: &PyModule) -> PyResult<()> {
    let core = PyModule::new(_py, "core")?;
    core.add_function(wrap_pyfunction!(open_array, core)?)?;
    m.add_submodule(core);
    Ok(())
}
