use pyo3::{exceptions::PyTypeError, PyErr};
use zarrs::array_subset::ArraySubset;

pub fn err<T>(msg: String) -> Result<T, PyErr> { Err(PyErr::new::<PyTypeError, _>(msg)) }

pub fn update_bytes_flen(
    output_bytes: &mut [u8],
    output_shape: &[u64],
    subset_bytes: &[u8],
    subset: &ArraySubset,
    data_type_size: usize,
) {
    debug_assert_eq!(
        output_bytes.len(),
        usize::try_from(output_shape.iter().product::<u64>()).unwrap() * data_type_size
    );
    debug_assert_eq!(
        subset_bytes.len(),
        subset.num_elements_usize() * data_type_size,
    );

    let contiguous_indices =
        unsafe { subset.contiguous_linearised_indices_unchecked(output_shape) };
    let length = contiguous_indices.contiguous_elements_usize() * data_type_size;
    let mut decoded_offset = 0;
    // TODO: Par iteration?
    for (array_subset_element_index, _num_elements) in &contiguous_indices {
        let output_offset = usize::try_from(array_subset_element_index).unwrap() * data_type_size;
        debug_assert!((output_offset + length) <= output_bytes.len());
        debug_assert!((decoded_offset + length) <= subset_bytes.len());
        output_bytes[output_offset..output_offset + length]
            .copy_from_slice(&subset_bytes[decoded_offset..decoded_offset + length]);
        decoded_offset += length;
    }
}