# `zarrs-python`

```{warning}
The version of `zarr-python` we currently depend on is still in pre-release and this package is accordingly extremely experimental.  We cannot guarantee any stability or correctness at the moment, although we have tried to do extensive testing and make clear what we think we support and do not.
```

This project serves as a bridge between [`zarrs`](https://docs.rs/zarrs/latest/zarrs/) and [`zarr`](https://zarr.readthedocs.io/en/latest/index.html) via [`PyO3`](https://pyo3.rs/v0.22.3/).  The main goal of the project is to speed up i/o.

To use the project, simply install our package (which depends on `zarr-python>3.0.0b0`), and run:

```python
import zarr
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
```

You can then use your `zarr` as normal (with some caveats)!

## API

We export a `ZarrsCodecPipeline` class so that `zarr-python` can use the class but it is not meant to be instantiated and we do not guarantee the stability of its API beyond what is required so that `zarr-python` can use it.  Therefore, it is not documented here.  We also export two errors, `DiscontiguousArrayError` and `CollapsedDimensionError` that can be thrown in the process of converting to indexers that `zarrs` can understand (see below for more details).

There are two ways to control the concurrency of the i/o **TODO: Need to clarify this**

## Supported Indexing Methods

We **do not** officially support the following indexing methods.  Some of these methods may error out, others may not:

1. Any discontinuous integer `np.ndarray` indexing for writes in 2D, and any integer `np.ndarray` indexing with dimensionality >=3 i.e.,
```python
arr[np.array([0, 5]), :] = ...
arr[np.array([...]), np.array([...]),  np.array([...])]
arr[np.array([...]), np.array([...]),  np.array([...])] = ...
```
2. `vindex` writes in 2D where both indexers are integer `np.ndarray` indices i.e.,
```python
arr[np.array([...]), np.array([...])] = ...
```
3. Ellipsis indexing.  We have tested some, but others fail even with `zarr-python`'s default codec pipeline.  Thus for now we advise proceeding with cuation here.
```python
arr[0:10, ..., 0:5]
```

Otherwise, we believe that we support your indexing case: slices, ints, and all integer `np.ndarray` indices in 2D for reading, contiguous integer `np.ndarray` indices along one axis for writing etc.  Please file an issue if you believe we have more holes in our coverage than we are aware of or you wish to contribute!  For example, we have an [issue in zarrs for integer-array indexing](https://github.com/LDeakin/zarrs/issues/52) that would unblock a lot of these issues!

That being said, using non-contiguous integer `np.ndarray` indexing for reads may not be as fast as expected given the performance of other supported methods.  Until `zarrs` supports integer indexing, only fetching chunks is done in `rust` while indexing then occurs in `python`.
