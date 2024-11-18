# Contributing

## Rust

You will need `rust` and `cargo` installed on your local system.  For more info, see [the rust docs](https://doc.rust-lang.org/cargo/getting-started/installation.html).

## Environment management

We encourage the use of [uv](https://docs.astral.sh/uv/) for environment management.  To install the package for development, run

```shell
uv pip install -e ".[test,dev,doc]"
```

However, take note that while this does build the rust package, the rust package will not be rebuilt upon edits despite the `-e` flag.  You will need to manually rebuild it using either `uv pip install -e .` or `maturin develop`.  Take note that for benchmarking/speed testing, it is advisable to build a release version of the rust package by passing the `-r` flag to `maturin`.  For more information on the `rust`-`python` bridge, see the [`PyO3` docs](https://pyo3.rs/v0.22.6/).

## Testing

To install test dependencies, simply run

```shell
pytest
```

or

```shell
pytest -n auto
```

for parallelized tests.  Most tests have been copied from the `zarr-python` repository with the exception of `test_pipeline.py` which we have written.

## Type hints

When authoring Python code, your IDE will not be able to analyze the extension module `zarrs._internal`.
But thanks to [`pyo3-stub-gen`][], we can generate type stubs for it!

To build the stub generator, run `cargo build --bin stub_gen`.
Afterwards, whenever you `cargo build`, `maturin build` or interact with your editor’s rust language server (e.g. `rust-analyzer`), the type hints will be updated.

[`pyo3-stub-gen`]: https://github.com/Jij-Inc/pyo3-stub-gen
