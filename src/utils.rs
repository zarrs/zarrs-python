use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::{Bound, PyResult};

use crate::{ChunksItem, WithSubset};

pub(crate) trait PyUntypedArrayExt {
    fn shape_zarr(&self) -> PyResult<Vec<u64>>;
}

impl PyUntypedArrayExt for Bound<'_, PyUntypedArray> {
    fn shape_zarr(&self) -> PyResult<Vec<u64>> {
        Ok(if self.shape().is_empty() {
            vec![1] // scalar value
        } else {
            self.shape()
                .iter()
                .map(|&i| u64::try_from(i))
                .collect::<Result<_, _>>()?
        })
    }
}

pub fn is_whole_chunk(item: &WithSubset) -> bool {
    item.chunk_subset.start().iter().all(|&o| o == 0)
        && item.chunk_subset.shape() == item.representation().shape_u64()
}
