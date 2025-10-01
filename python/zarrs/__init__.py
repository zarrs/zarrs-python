from ._internal import __version__
from .pipeline import ZarrsCodecPipeline as _ZarrsCodecPipeline
from .utils import CollapsedDimensionError, DiscontiguousArrayError


# Need to do this redirection so people can access the pipeline as `zarrs.ZarrsCodecPipeline` instead of `zarrs.pipeline.ZarrsCodecPipeline`
class ZarrsCodecPipeline(_ZarrsCodecPipeline):
    pass


__all__ = [
    "ZarrsCodecPipeline",
    "DiscontiguousArrayError",
    "CollapsedDimensionError",
    "__version__",
]
