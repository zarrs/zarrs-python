import zarrs_python # noqa: F401
import pytest
import os


@pytest.fixture(autouse=True)
def run_around_each_test():
    os.environ["ZARR_CODEC_PIPELINE__PATH"] = "zarrs_python.ZarrsCodecPipeline"
    yield
    os.environ["ZARR_CODEC_PIPELINE__PATH"] = "zarrs_python.ZarrsCodecPipeline"

@pytest.fixture(scope="session")
def run_before():
    os.environ["ZARR_CODEC_PIPELINE__PATH"] = "zarrs_python.ZarrsCodecPipeline"
    yield
    os.environ["ZARR_CODEC_PIPELINE__PATH"] = "zarrs_python.ZarrsCodecPipeline"