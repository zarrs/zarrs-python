# `zarrs-python`

```{warning}
The version of `zarr-python` we currently depend on is still in pre-release and this package is accordingly extremely experimental.  We cannot guarantee any stability or correctness at the moment, although we have tried to do extensive testing and make clear what we think we support and do not.
```

This project serves a python bridge between [`zarrs`](https://docs.rs/zarrs/latest/zarrs/) and `python` via [`PyO3`](https://pyo3.rs/v0.22.3/).  The main goal of the project is to speed up i/o of zarr datasets.

To use the project, simply install our package (which depends on `zarr-python>3.0.0b0`), and run:

```python
import zarr
zarr.config.set({"codec_pipeline.path": "zarrs_python.ZarrsCodecPipeline"})
```

You can then use your `zarr` arrays as normal!

## API

We export a `ZarrsCodecPipeline` class so that `zarr-python` can use the class but it is not meant to be instantiated and we do not guarantee the stability of its API beyond what is required so that `zarr-python` can use it.  Therefore, it is not documented here.  We also export two errors, `DiscontiguousArrayError` and `CollapsedDimensionError` that can be thrown in the process of converting to indexers that `zarrs` can understand (see below for more details).

There are two ways to control the concurrency of the i/o **TODO: Need to clarify this**

## Supported Indexing Methods

We **do not** officially support the following indexing methods.  Some of these methods may error out, others may not:

1. Any discontinuous integer indexing for writes in 2D, and any integer-based indexing with dimensionality >=3
2. `vindex` writes in 2D where both indexers are integer indices
3. Ellipsis indexing.  We have tested some, but others fail even with `zarr-python`'s default codec pipeline.  Thus for now we advise proceeding with cuation here.

Otherwise, we believe that we support your indexing case: slices, ints, and all integer indices in 2D for reading, contiguous integer indices along one axis for writing etc.  Please file an issue if you believe we have more holes in our coverage than we are aware of or you wish to contribute!  For example, https://github.com/LDeakin/zarrs/issues/52 would unblock a lot of these issues!