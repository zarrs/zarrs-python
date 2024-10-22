use pyo3::{exceptions::PyTypeError, PyErr};

pub fn err<T>(msg: String) -> Result<T, PyErr> {
    Err(PyErr::new::<PyTypeError, _>(msg))
}
