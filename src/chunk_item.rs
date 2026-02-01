use std::num::NonZeroU64;

use pyo3::{
    Bound, PyAny, PyErr, PyResult,
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::{PyAnyMethods, PyBytes, PyBytesMethods, PyInt, PySlice, PySliceMethods as _},
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zarrs::{
    array::{ArraySubset, ChunkShape, DataType, FillValue},
    metadata::v3::MetadataV3,
    storage::StoreKey,
};

use crate::utils::PyErrExt;

pub(crate) trait ChunksItem {
    fn key(&self) -> &StoreKey;
    fn shape(&self) -> &[NonZeroU64];
    fn data_type(&self) -> &DataType;
    fn fill_value(&self) -> &FillValue;
}

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct Basic {
    key: StoreKey,
    shape: ChunkShape,
    data_type: DataType,
    fill_value: FillValue,
}

fn fill_value_to_bytes(dtype: &str, fill_value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
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

#[gen_stub_pymethods]
#[pymethods]
impl Basic {
    #[new]
    fn new(byte_interface: &Bound<'_, PyAny>, chunk_spec: &Bound<'_, PyAny>) -> PyResult<Self> {
        let path: String = byte_interface.getattr("path")?.extract()?;

        let shape: Vec<NonZeroU64> = chunk_spec.getattr("shape")?.extract()?;

        let mut dtype: String = chunk_spec
            .getattr("dtype")?
            .call_method0("to_native_dtype")?
            .call_method0("__str__")?
            .extract()?;
        if dtype == "object" {
            // zarrs doesn't understand `object` which is the output of `np.dtype("|O").__str__()`
            // but maps it to "string" internally https://github.com/LDeakin/zarrs/blob/0532fe983b7b42b59dbf84e50a2fe5e6f7bad4ce/zarrs_metadata/src/v2_to_v3.rs#L288
            dtype = String::from("string");
        }
        let data_type = get_data_type_from_dtype(&dtype)?;
        let fill_value: Bound<'_, PyAny> = chunk_spec.getattr("fill_value")?;
        let fill_value = FillValue::new(fill_value_to_bytes(&dtype, &fill_value)?);
        Ok(Self {
            key: StoreKey::new(path).map_py_err::<PyValueError>()?,
            shape,
            data_type,
            fill_value,
        })
    }
}

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct WithSubset {
    pub item: Basic,
    pub chunk_subset: ArraySubset,
    pub subset: ArraySubset,
}

#[gen_stub_pymethods]
#[pymethods]
impl WithSubset {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        item: Basic,
        chunk_subset: Vec<Bound<'_, PySlice>>,
        subset: Vec<Bound<'_, PySlice>>,
        shape: Vec<u64>,
    ) -> PyResult<Self> {
        let chunk_subset = selection_to_array_subset(&chunk_subset, &item.shape)?;
        let shape: Vec<NonZeroU64> = shape
            .into_iter()
            .map(|dim| {
                NonZeroU64::new(dim)
                    .ok_or("subset dimensions must be greater than zero")
                    .map_py_err::<PyValueError>()
            })
            .collect::<PyResult<Vec<NonZeroU64>>>()?;
        let subset = selection_to_array_subset(&subset, &shape)?;
        // Check that subset and chunk_subset have the same number of elements.
        // This permits broadcasting of a constant input.
        if subset.num_elements() != chunk_subset.num_elements() && subset.num_elements() > 1 {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "the size of the chunk subset {chunk_subset} and input/output subset {subset} are incompatible",
            )));
        }
        Ok(Self {
            item,
            chunk_subset,
            subset,
        })
    }
}

impl ChunksItem for Basic {
    fn key(&self) -> &StoreKey {
        &self.key
    }
    fn shape(&self) -> &[NonZeroU64] {
        &self.shape
    }
    fn data_type(&self) -> &DataType {
        &self.data_type
    }
    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }
}

impl ChunksItem for WithSubset {
    fn key(&self) -> &StoreKey {
        &self.item.key
    }
    fn shape(&self) -> &[NonZeroU64] {
        &self.item.shape
    }
    fn data_type(&self) -> &DataType {
        &self.item.data_type
    }
    fn fill_value(&self) -> &FillValue {
        &self.item.fill_value
    }
}

fn get_data_type_from_dtype(dtype: &str) -> PyResult<DataType> {
    let data_type =
        DataType::from_metadata(&MetadataV3::new(dtype)).map_py_err::<PyRuntimeError>()?;
    Ok(data_type)
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
