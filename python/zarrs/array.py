from __future__ import annotations

import numpy as np
import zarr
from zarr.core.array import Array

from ._internal import ArrayImpl


def _is_basic_indexing(key) -> bool:
    """Check if key uses only int, step-1 slices, and/or a single Ellipsis."""
    if not isinstance(key, tuple):
        key = (key,)
    has_ellipsis = False
    for k in key:
        if isinstance(k, int):
            continue
        elif isinstance(k, slice):
            if k.step is not None and k.step != 1:
                return False
        elif k is Ellipsis:
            if has_ellipsis:
                return False  # multiple ellipses
            has_ellipsis = True
        else:
            return False
    return True


class _LazySlice:
    """Lazy reference to a subset of a ZarrsArray (no I/O until consumed)."""

    __slots__ = ("_dtype", "_impl", "_ranges", "_region_shape", "_squeeze_dims")

    def __init__(self, impl_, ranges, region_shape, dtype, squeeze_dims):
        self._impl = impl_
        self._ranges = ranges
        self._region_shape = region_shape
        self._dtype = dtype
        self._squeeze_dims = squeeze_dims

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out = np.empty(self._region_shape, dtype=self._dtype)
        if out.size > 0:
            self._impl.retrieve(self._ranges, out)
        if self._squeeze_dims:
            out = out.squeeze(axis=tuple(self._squeeze_dims))
        if dtype is not None and out.dtype != dtype:
            out = out.astype(dtype, copy=False)
        return out


class _LazyIndexer:
    """Proxy returned by ``ZarrsArray.lazy`` that captures indexing lazily."""

    __slots__ = ("_pipeline",)

    def __init__(self, pipeline: ZarrsArray):
        self._pipeline = pipeline

    def __getitem__(self, key: slice | int | tuple[slice | int, ...]) -> _LazySlice:
        ranges, region_shape, squeeze_dims = self._pipeline._parse_key(key)
        return _LazySlice(
            self._pipeline._impl,
            ranges,
            region_shape,
            self._pipeline.dtype,
            squeeze_dims,
        )


class ZarrsArray(Array):
    """zarr.Array subclass backed by zarrs for fast I/O.

    Supports all zarr.Array operations. Basic slice indexing (ints, step-1
    slices, ellipsis) is handled by the Rust fast path; advanced indexing
    falls back to zarr.Array unless ``codec_pipeline.strict`` is set.
    """

    def __init__(
        self,
        array: Array,
        *,
        validate_checksums: bool = False,
        chunk_concurrent_minimum: int | None = None,
        num_threads: int | None = None,
        direct_io: bool = False,
    ) -> None:
        super().__init__(array._async_array)
        store = array.store_path.store
        zarr_path = array.store_path.path
        zarrs_path = "/" + zarr_path if zarr_path else "/"
        self._impl = ArrayImpl(
            store,
            zarrs_path,
            validate_checksums=validate_checksums,
            chunk_concurrent_minimum=chunk_concurrent_minimum,
            num_threads=num_threads,
            direct_io=direct_io,
        )

    @property
    def lazy(self) -> _LazyIndexer:
        return _LazyIndexer(self)

    def _parse_key(
        self, key: slice | int | tuple[slice | int, ...]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not isinstance(key, tuple):
            key = (key,)

        # Expand Ellipsis
        if Ellipsis in key:
            idx = key.index(Ellipsis)
            n_explicit = len(key) - 1  # everything except the Ellipsis
            n_expand = self.ndim - n_explicit
            if n_expand < 0:
                raise IndexError(
                    f"too many indices for array: "
                    f"array is {self.ndim}-dimensional, "
                    f"but {n_explicit} were indexed"
                )
            key = key[:idx] + (slice(None),) * n_expand + key[idx + 1 :]

        if len(key) > self.ndim:
            raise IndexError(
                f"too many indices for array: "
                f"array is {self.ndim}-dimensional, "
                f"but {len(key)} were indexed"
            )

        # Pad missing dimensions with full slices
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))

        ranges: list[tuple[int, int]] = []
        region_shape: list[int] = []
        squeeze_dims: list[int] = []

        for i, (k, dim_size) in enumerate(zip(key, self.shape)):
            if isinstance(k, int):
                if k < 0:
                    k += dim_size
                if k < 0 or k >= dim_size:
                    raise IndexError(
                        f"index {k} is out of bounds for axis {i} with size {dim_size}"
                    )
                ranges.append((k, k + 1))
                region_shape.append(1)
                squeeze_dims.append(i)
            elif isinstance(k, slice):
                start, stop, step = k.indices(dim_size)
                if step != 1:
                    raise IndexError("only step=1 slices are supported")
                ranges.append((start, stop))
                region_shape.append(max(0, stop - start))
            else:
                raise IndexError(f"unsupported index type: {type(k).__name__}")

        return ranges, region_shape, squeeze_dims

    def __getitem__(self, key: slice | int | tuple[slice | int, ...]) -> np.ndarray:
        if _is_basic_indexing(key):
            ranges, region_shape, squeeze_dims = self._parse_key(key)
            out = np.empty(region_shape, dtype=self.dtype)
            if out.size > 0:
                self._impl.retrieve(ranges, out)
            if squeeze_dims:
                out = out.squeeze(axis=tuple(squeeze_dims))
            return out

        strict = zarr.config.get("codec_pipeline.strict", False)
        if strict:
            raise IndexError(
                "ZarrsArray in strict mode does not support advanced indexing"
            )
        return super().__getitem__(key)

    def __setitem__(self, key: slice | int | tuple[slice | int, ...], value) -> None:
        if _is_basic_indexing(key):
            ranges, region_shape, squeeze_dims = self._parse_key(key)

            if isinstance(value, _LazySlice):
                if value._region_shape != region_shape:
                    raise ValueError(
                        f"could not broadcast input array from shape "
                        f"{tuple(value._region_shape)} "
                        f"into shape {tuple(region_shape)}"
                    )
                if all(s > 0 for s in region_shape):
                    self._impl.copy_from(value._impl, value._ranges, ranges)
                return

            value = np.asarray(value, dtype=self.dtype)

            # Ensure native byte order
            if not value.dtype.isnative:
                value = value.byteswap().view(value.dtype.newbyteorder("="))

            # Expand squeezed dimensions back
            for dim in squeeze_dims:
                value = np.expand_dims(value, axis=dim)

            if value.shape != tuple(region_shape):
                raise ValueError(
                    f"could not broadcast input array from shape {value.shape} "
                    f"into shape {tuple(region_shape)}"
                )

            # Ensure C-contiguous before passing to Rust
            value = np.ascontiguousarray(value)

            if value.size > 0:
                self._impl.store(ranges, value)
            return

        strict = zarr.config.get("codec_pipeline.strict", False)
        if strict:
            raise IndexError(
                "ZarrsArray in strict mode does not support advanced indexing"
            )
        super().__setitem__(key, value)
