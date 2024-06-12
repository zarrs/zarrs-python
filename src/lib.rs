use pyo3::{exceptions::PyTypeError, prelude::*};
use std::sync::Arc;
use zarrs::storage::{ReadableStorage, store};
use zarrs::array::Array as RustArray;

mod array;
mod utils;

#[pyfunction]
fn open_array(path: &str) -> PyResult<array::Array> {
    let s: ReadableStorage = Arc::new(store::HTTPStore::new(path).or_else(|x| utils::err(x.to_string()))?);
    let arr  = RustArray::new(s, &"/").or_else(|x| utils::err(x.to_string()))?; 
    Ok(array::Array{ arr })
}

/// A Python module implemented in Rust.
#[pymodule]
fn zarrs_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let core = PyModule::new_bound(m.py(),"core")?;
    core.add_function(wrap_pyfunction!(open_array, &core)?)?;
    let _ = m.add_submodule(&core);
    Ok(())
}
