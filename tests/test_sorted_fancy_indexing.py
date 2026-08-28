"""Sorted integer-array reads reach the zarrs pipeline instead of falling back.

`strict` makes these categorical rather than merely correct: with no fallback, a selection
zarrs cannot serve raises instead of quietly returning the right answer via zarr-python's
pipeline. It must be set before the array is opened -- that is when the pipeline decides
whether it has a fallback -- so every test opens its own handle via `open_strict`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from zarrs.utils import DiscontiguousArrayError, UnsupportedVIndexingError

if TYPE_CHECKING:
    from pathlib import Path

SHAPE = (32, 24)
CHUNKS = (8, 6)
SHARDS = (16, 12)
ZARRS = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


def _write(path, values, chunks, shards) -> Path:
    zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=chunks, shards=shards
    )[:] = values
    return path


@pytest.fixture
def sharded(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
    return _write(tmp_path / "2d.zarr", values, CHUNKS, SHARDS), values


@pytest.fixture
def sharded_1d(tmp_path: Path) -> tuple[Path, np.ndarray]:
    values = np.arange(64, dtype=np.float64)
    return _write(tmp_path / "1d.zarr", values, (8,), (16,)), values


def open_strict(path: Path) -> zarr.Array:
    """Open with no fallback, so an unsupported selection raises instead of being rerouted."""
    with zarr.config.set({**ZARRS, "codec_pipeline.strict": True}):
        return zarr.open_array(path, mode="r+")


# Each case spans chunk *and* shard boundaries, and mixes runs of length 1 with longer ones.
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(np.array([0, 3, 4, 5, 17, 30]), id="rows"),
        pytest.param((np.array([2, 3, 20]), slice(4, 18)), id="rows-slice"),
        pytest.param((slice(None), np.array([0, 1, 7, 23])), id="cols"),
        # Do not drop: on main this raises `RuntimeError: incompatible offset`, which the
        # fallback does not catch, so it is the one case here fixing a hard crash.
        pytest.param((slice(6, 9), np.array([5, 6, 13])), id="slice-cols"),
        pytest.param((np.array([4, 5, 6, 29]), 7), id="rows-int"),
        pytest.param(np.array([0, 31]), id="rows-endpoints"),
        pytest.param(np.array([11]), id="single-row"),
        pytest.param(np.arange(0, 32), id="every-row"),
        # A repeat ends its run early and reads that index again into the next output slot.
        pytest.param(np.array([3, 3, 3]), id="all-repeats"),
        pytest.param(np.array([0, 0, 1, 2, 2]), id="repeats-either-side-of-a-run"),
    ],
)
def test_sorted_integer_array_read(
    sharded: tuple[Path, np.ndarray], index: object
) -> None:
    path, expected = sharded
    np.testing.assert_array_equal(open_strict(path)[index], expected[index])


def test_sorted_vindex_1d(sharded_1d: tuple[Path, np.ndarray]) -> None:
    path, expected = sharded_1d
    index = np.array([0, 1, 5, 12, 13, 14, 63])
    z = open_strict(path)
    np.testing.assert_array_equal(z.vindex[index], expected[index])
    np.testing.assert_array_equal(z[index], expected[index])


# Indices within one shard, so a single chunk item really does get several of them -- spread
# across shards each item gets one index, which is a box and was always supported.
@pytest.mark.parametrize(
    "index",
    [
        # Unsorted: zarr-python reorders the output, so a run's position in the selection is
        # not its position in the output.
        pytest.param(np.array([9, 2]), id="unsorted-rows"),
        # Two array axes: outer and coordinate indexing disagree on what this means.
        pytest.param((np.array([1, 3]), np.array([0, 2])), id="two-array-axes"),
        pytest.param((slice(None), slice(None, None, 2)), id="strided"),
    ],
)
def test_unsupported_raises_strictly_but_falls_back_correctly(
    sharded: tuple[Path, np.ndarray], index: object
) -> None:
    path, expected = sharded
    with pytest.raises((DiscontiguousArrayError, UnsupportedVIndexingError)):
        open_strict(path)[index]
    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode="r")
        np.testing.assert_array_equal(z[index], expected[index])


def test_writes_are_not_split(
    sharded: tuple[Path, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A split write would make several read-modify-writes of one chunk race.

    The values alone prove nothing: without strict mode the write falls back to zarr-python
    whatever happens, so this passed identically when writes WERE split. What it asserts is
    that the splitter is never reached. Rows 1, 3 and 4 share a chunk -- the losing case.
    """
    path, expected = sharded
    index = np.array([1, 3, 4])
    monkeypatch.setattr(
        "zarrs.utils.split_selection_runs",
        lambda *_: pytest.fail("a write reached split_selection_runs"),
    )
    with zarr.config.set(ZARRS):
        z = zarr.open_array(path, mode="r+")
        z[index, :] = np.full((len(index), SHAPE[1]), -1.0)

        # Undone before reading back: a READ is meant to reach the splitter.
        monkeypatch.undo()
        expected[index, :] = -1.0
        np.testing.assert_array_equal(z[...], expected)


def test_contiguous_output_does_not_imply_sorted_input(
    sharded_1d: tuple[Path, np.ndarray],
) -> None:
    """A rectangular output side is not evidence the input was ordered.

    `CoordinateIndexer` sorts only when the chunk-raveled order is wrong, and 7 and 3 both
    live in chunk 0, so `out_selection` comes back `slice(0, 2)` while the indices descend.
    Building runs from that would give the inverted box `slice(7, 4)`.
    """
    path, expected = sharded_1d
    index = np.array([7, 3])
    with pytest.raises((DiscontiguousArrayError, UnsupportedVIndexingError)):
        open_strict(path).vindex[index]
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r").vindex[index]
    np.testing.assert_array_equal(got, expected[index])


def test_a_split_read_is_rejected_without_the_splitter(
    sharded: tuple[Path, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tests above must be exercising the splitter, not passing for some other reason."""
    monkeypatch.setattr(
        "zarrs.utils.split_selection_runs",
        lambda chunk_sel, out_sel, chunk_shape=None: ((chunk_sel, out_sel),),
    )
    path, _ = sharded
    with pytest.raises(DiscontiguousArrayError):
        open_strict(path)[np.array([0, 3, 4, 5, 17, 30])]


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param(np.arange(SHAPE[0]) < 16, id="aligned-to-chunks"),
        pytest.param(np.ones(SHAPE[0], dtype=bool), id="every-row"),
        pytest.param(np.isin(np.arange(SHAPE[0]), [3, 17, 30]), id="scattered"),
        pytest.param(np.zeros(SHAPE[0], dtype=bool), id="no-rows"),
    ],
)
def test_boolean_mask_reads_the_positions_it_marks(
    sharded: tuple[Path, np.ndarray], mask: np.ndarray
) -> None:
    """A mask is not an index array, and casting one is silently wrong.

    `BoolArrayDimIndexer` hands over a BOOLEAN chunk selection with a slice out-selection.
    Cast to int64 it becomes [1, 1, ...] -- non-decreasing, and exactly as long as the
    output slice -- so it passes every eligibility test and reads element 1 once per True.
    Values only, no exception. Masks aligned to chunk boundaries hit it every time.
    """
    path, expected = sharded
    with zarr.config.set(ZARRS):
        got = zarr.open_array(path, mode="r")[mask, :]
    np.testing.assert_array_equal(got, expected[mask, :])
