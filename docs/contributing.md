# Contributing

## Rust

You will need `Rust` and `cargo` installed on your local system.
For more info, see [the Rust docs](https://doc.rust-lang.org/cargo/getting-started/installation.html).

## Environment management

We encourage the use of [`uv`](https://docs.astral.sh/uv/) for environment management.
To create a virtual environment for development with all `dev` dependencies, run

```shell
uv sync
```

If you prefer not to use `uv`, simply create and activate a virtual environment, then run:
```shell
pip install --group dev -e .
```

> [!NOTE]
> The rest of this document assumes you are using `uv`.
> If you are not using `uv`, just remove the `uv run` preceeding commands.

> [!WARNING]
> The above commands initially build the `Rust` package, but it will not be rebuilt upon edits.
> You will need to manually rebuild it using `uv run maturin develop`.

For more information on the `Rust`-`Python` bridge, see the [`PyO3` docs](https://pyo3.rs/v0.22.6/).

## Testing

To run tests

```shell
uv run maturin develop
uv run pytest -n auto
```

Most tests have been copied from the `zarr-python` repository with the exception of `test_pipeline.py` which we have written.

For benchmarking/speed testing, it is advisable to build a release version of the `Rust` package by passing the `-r` flag to `maturin`.

## Type hints

Thanks to [`pyo3-stub-gen`][], we can generate type stubs for the `zarrs._internal` module.
If the “Check formatting” CI step fails, run the following and commit the changes:
```shell
cargo run --bin stub_gen
uv run pre-commit run --all-files
```

Once `maturin` can be run as a `hatchling` plugin, this can be made automatic.

[`pyo3-stub-gen`]: https://github.com/Jij-Inc/pyo3-stub-gen
