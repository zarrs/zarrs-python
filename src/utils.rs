use pyo3::{exceptions::PyTypeError, PyErr};
use zarrs::array_subset::ArraySubset;

pub fn err<T>(msg: String) -> Result<T, PyErr> {
    Err(PyErr::new::<PyTypeError, _>(msg))
}
