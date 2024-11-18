from importlib.metadata import version

import zarrs


def test_version():
    assert zarrs.__version__ == version("zarrs")
