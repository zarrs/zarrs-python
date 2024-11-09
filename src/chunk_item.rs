use std::{num::NonZeroU64, sync::Arc};

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    types::{PySlice, PySliceMethods},
    Bound, PyErr, PyResult,
};
use zarrs::{
    array::{ChunkRepresentation, DataType, FillValue},
    array_subset::ArraySubset,
    metadata::v3::{array::data_type::DataTypeMetadataV3, MetadataV3},
    storage::{ReadableWritableListableStorageTraits, StoreKey},
};

use crate::utils::PyErrExt;

pub(crate) type ChunksItemRaw<'a> = (
    // store path
    String,
    // shape
    Vec<u64>,
    // data type
    String,
    // fill value bytes
    Vec<u8>,
);

pub(crate) type ChunksItemRawWithIndices<'a> = (
    ChunksItemRaw<'a>,
    // out selection
    Vec<Bound<'a, PySlice>>,
    // chunk selection
    Vec<Bound<'a, PySlice>>,
);

pub(crate) trait IntoItem<T>: std::marker::Sized {
    fn store_path(&self) -> &str;
    fn into_item(
        self,
        store: Arc<dyn ReadableWritableListableStorageTraits>,
        key: StoreKey,
        shape: &[u64],
    ) -> PyResult<T>;
}

pub(crate) struct ChunksItem {
    pub store: Arc<dyn ReadableWritableListableStorageTraits>,
    pub key: StoreKey,
    pub representation: ChunkRepresentation,
}

pub(crate) struct ChunksItemWithSubset {
    pub item: ChunksItem,
    pub chunk_subset: ArraySubset,
    pub subset: ArraySubset,
}

impl<'a> IntoItem<ChunksItem> for ChunksItemRaw<'a> {
    fn store_path(&self) -> &str {
        &self.0
    }
    fn into_item(
        self,
        store: Arc<dyn ReadableWritableListableStorageTraits>,
        key: StoreKey,
        _: &[u64],
    ) -> PyResult<ChunksItem> {
        let (_, chunk_shape, dtype, fill_value) = self;
        let representation = get_chunk_representation(chunk_shape, &dtype, fill_value)?;
        Ok(ChunksItem {
            store,
            key,
            representation,
        })
    }
}

impl IntoItem<ChunksItemWithSubset> for ChunksItemRawWithIndices<'_> {
    fn store_path(&self) -> &str {
        &self.0 .0
    }
    fn into_item(
        self,
        store: Arc<dyn ReadableWritableListableStorageTraits>,
        key: StoreKey,
        shape: &[u64],
    ) -> PyResult<ChunksItemWithSubset> {
        let (raw, selection, chunk_selection) = self;
        let chunk_shape = raw.1.clone();
        let item = raw.into_item(store.clone(), key, shape)?;
        Ok(ChunksItemWithSubset {
            item,
            chunk_subset: selection_to_array_subset(&chunk_selection, &chunk_shape)?,
            subset: selection_to_array_subset(&selection, shape)?,
        })
    }
}

fn get_chunk_representation(
    chunk_shape: Vec<u64>,
    dtype: &str,
    fill_value: Vec<u8>,
) -> PyResult<ChunkRepresentation> {
    // Get the chunk representation
    let data_type =
        DataType::from_metadata(&DataTypeMetadataV3::from_metadata(&MetadataV3::new(dtype)))
            .map_py_err::<PyRuntimeError>()?;
    let chunk_shape = chunk_shape
        .into_iter()
        .map(|x| NonZeroU64::new(x).expect("chunk shapes should always be non-zero"))
        .collect();
    let chunk_representation =
        ChunkRepresentation::new(chunk_shape, data_type, FillValue::new(fill_value))
            .map_py_err::<PyValueError>()?;
    Ok(chunk_representation)
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
    shape: &[u64],
) -> PyResult<ArraySubset> {
    if selection.is_empty() {
        Ok(ArraySubset::new_with_shape(vec![1; shape.len()]))
    } else {
        let chunk_ranges = selection
            .iter()
            .zip(shape)
            .map(|(selection, &shape)| slice_to_range(selection, isize::try_from(shape)?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(ArraySubset::new_with_ranges(&chunk_ranges))
    }
}
