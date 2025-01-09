from zarr.registry import register_pipeline

from ._internal import __version__
from .pipeline import ZarrsCodecPipeline as ZarrsCodecPipeline
from .utils import CollapsedDimensionError, DiscontiguousArrayError

register_pipeline(ZarrsCodecPipeline)

__all__ = [
    "ZarrsCodecPipeline",
    "DiscontiguousArrayError",
    "CollapsedDimensionError",
    "__version__",
]
