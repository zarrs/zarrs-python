use std::num::NonZeroU64;

use pyo3::{
    Bound, PyAny, PyErr, PyResult,
    exceptions::{PyIndexError, PyValueError},
    pyclass, pymethods,
    types::{PyAnyMethods, PyBytes, PyBytesMethods, PyInt, PySlice, PySliceMethods as _},
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zarrs::{array_subset::ArraySubset, storage::StoreKey};

use crate::utils::PyErrExt;

pub fn fill_value_to_bytes(dtype: &str, fill_value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if dtype == "string" {
        // Match zarr-python 2.x.x string fill value behaviour with a 0 fill value
        // See https://github.com/zarr-developers/zarr-python/issues/2792#issuecomment-2644362122
        if let Ok(fill_value_downcast) = fill_value.cast::<PyInt>() {
            let fill_value_usize: usize = fill_value_downcast.extract()?;
            if fill_value_usize == 0 {
                return Ok(vec![]);
            }
            Err(PyErr::new::<PyValueError, _>(format!(
                "Cannot understand non-zero integer {fill_value_usize} fill value for dtype {dtype}"
            )))?;
        }
    }

    if let Ok(fill_value_downcast) = fill_value.cast::<PyBytes>() {
        Ok(fill_value_downcast.as_bytes().to_vec())
    } else if fill_value.hasattr("tobytes")? {
        Ok(fill_value.call_method0("tobytes")?.extract()?)
    } else {
        Err(PyErr::new::<PyValueError, _>(format!(
            "Unsupported fill value {fill_value:?}"
        )))
    }
}

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct WithSubset {
    key: StoreKey,
    pub chunk_subset: ArraySubset,
    pub subset: ArraySubset,
    shape: Vec<NonZeroU64>,
    pub num_elements: u64,
}

#[gen_stub_pymethods]
#[pymethods]
impl WithSubset {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        key: String,
        chunk_subset: Vec<Bound<'_, PySlice>>,
        chunk_shape: Vec<u64>,
        subset: Vec<Bound<'_, PySlice>>,
        shape: Vec<u64>,
    ) -> PyResult<Self> {
        let num_elements = chunk_shape.iter().product();
        let shape_nonzero_u64: Vec<NonZeroU64> = shape
            .into_iter()
            .map(|dim| {
                NonZeroU64::new(dim).ok_or_else(|| {
                    PyErr::new::<PyValueError, _>(
                        "subset dimensions must be greater than zero".to_string(),
                    )
                })
            })
            .collect::<PyResult<Vec<NonZeroU64>>>()?;
        let chunk_shape_nonzero_u64: Vec<NonZeroU64> = chunk_shape
            .into_iter()
            .map(|dim| {
                NonZeroU64::new(dim).ok_or_else(|| {
                    PyErr::new::<PyValueError, _>(
                        "subset dimensions must be greater than zero".to_string(),
                    )
                })
            })
            .collect::<PyResult<Vec<NonZeroU64>>>()?;
        let chunk_subset = selection_to_array_subset(&chunk_subset, &chunk_shape_nonzero_u64)?;
        let subset = selection_to_array_subset(&subset, &shape_nonzero_u64)?;
        // Check that subset and chunk_subset have the same number of elements.
        // This permits broadcasting of a constant input.
        if subset.num_elements() != chunk_subset.num_elements() && subset.num_elements() > 1 {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "the size of the chunk subset {chunk_subset} and input/output subset {subset} are incompatible",
            )));
        }

        Ok(Self {
            key: StoreKey::new(key).map_py_err::<PyValueError>()?,
            chunk_subset,
            subset,
            shape: chunk_shape_nonzero_u64,
            num_elements,
        })
    }
}
impl WithSubset {
    pub fn key(&self) -> &StoreKey {
        &self.key
    }
    pub fn shape(&self) -> &[NonZeroU64] {
        &self.shape
    }
}

fn slice_to_range(slice: &Bound<'_, PySlice>, length: isize) -> PyResult<std::ops::Range<u64>> {
    let indices = slice.indices(length)?;
    if indices.start < 0 {
        Err(PyErr::new::<PyValueError, _>(
            "slice start must be greater than or equal to 0".to_string(),
        ))
    } else if indices.stop < 0 {
        Err(PyErr::new::<PyValueError, _>(
            "slice stop must be greater than or equal to 0".to_string(),
        ))
    } else if indices.step != 1 {
        Err(PyErr::new::<PyValueError, _>(
            "slice step must be equal to 1".to_string(),
        ))
    } else {
        Ok(u64::try_from(indices.start)?..u64::try_from(indices.stop)?)
    }
}

fn selection_to_array_subset(
    selection: &[Bound<'_, PySlice>],
    shape: &[NonZeroU64],
) -> PyResult<ArraySubset> {
    if selection.is_empty() {
        Ok(ArraySubset::new_with_shape(vec![1; shape.len()]))
    } else {
        let chunk_ranges = selection
            .iter()
            .zip(shape)
            .map(|(selection, &shape)| slice_to_range(selection, isize::try_from(shape.get())?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(ArraySubset::new_with_ranges(&chunk_ranges))
    }
}
