#![warn(clippy::pedantic)]

use pyo3::prelude::*;
use std::sync::Arc;
use zarrs::array::Array as RustArray;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableStorage;
use zarrs_http::HTTPStore;

mod array;
mod utils;

#[pyfunction]
fn open_array(path: &str, n_threads: usize) -> PyResult<array::ZarrsPythonArray> {
    #![allow(deprecated)] // HTTPStore is moved to an independent crate in zarrs 0.17 and undeprecated
    let s: ReadableStorage = if path.starts_with("http://") | path.starts_with("https://") {
        Arc::new(HTTPStore::new(path).or_else(|x| utils::err(x.to_string()))?)
    } else {
        Arc::new(FilesystemStore::new(path).or_else(|x| utils::err(x.to_string()))?)
    };
    let arr = RustArray::open(s, "/").or_else(|x| utils::err(x.to_string()))?;
    let threadpool = rayon::ThreadPoolBuilder::new()
        .num_threads(if n_threads > 0 {
            n_threads
        } else {
            std::thread::available_parallelism().unwrap().get()
        })
        .build()
        .unwrap();
    Ok(array::ZarrsPythonArray::new(arr, threadpool))
}

/// A Python module implemented in Rust.
#[pymodule]
fn zarrs_python_internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open_array, m)?)?;
    m.add_class::<array::ZarrsPythonArray>()?;
    Ok(())
}
