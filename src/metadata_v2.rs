use pyo3::{exceptions::PyRuntimeError, pyfunction, PyErr, PyResult};
use zarrs::metadata::{
    v2::{array::ArrayMetadataV2Order, MetadataV2},
    v3::array::data_type::DataTypeMetadataV3,
};

#[pyfunction]
#[pyo3(signature = (filters=None, compressor=None))]
pub fn codec_metadata_v2_to_v3(
    filters: Option<Vec<String>>,
    compressor: Option<String>,
) -> PyResult<Vec<String>> {
    // Try and convert filters/compressor to V2 metadata
    let filters = if let Some(filters) = filters {
        Some(
            filters
                .into_iter()
                .map(|filter| {
                    serde_json::from_str::<MetadataV2>(&filter)
                        .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    let compressor = if let Some(compressor) = compressor {
        Some(
            serde_json::from_str::<MetadataV2>(&compressor)
                .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?,
        )
    } else {
        None
    };

    // FIXME: The array order, dimensionality, data type, and endianness are needed to exhaustively support all Zarr V2 data that zarrs can handle.
    // However, CodecPipeline.from_codecs does not supply this information, and CodecPipeline.evolve_from_array_spec is seemingly never called.
    let metadata = zarrs::metadata::v2_to_v3::codec_metadata_v2_to_v3(
        ArrayMetadataV2Order::C,
        0,                         // unused with C order
        &DataTypeMetadataV3::Bool, // FIXME
        None,
        &filters,
        &compressor,
    )
    .map_err(|err| {
        // TODO: More informative error messages from zarrs for ArrayMetadataV2ToV3ConversionError
        PyErr::new::<PyRuntimeError, _>(err.to_string())
    })?;
    Ok(metadata
        .into_iter()
        .map(|metadata| serde_json::to_string(&metadata).expect("infallible")) // TODO: Add method to zarrs
        .collect())
}
