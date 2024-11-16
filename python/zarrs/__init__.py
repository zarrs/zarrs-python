from importlib.metadata import version
from zarr.registry import register_pipeline

from .pipeline import ZarrsCodecPipeline as _ZarrsCodecPipeline
from .utils import CollapsedDimensionError, DiscontiguousArrayError

__version__ = version("zarrs")

# Need to do this redirection so people can access the pipeline as `zarrs.ZarrsCodecPipeline` instead of `zarrs.pipeline.ZarrsCodecPipeline`
class ZarrsCodecPipeline(_ZarrsCodecPipeline):
    pass


register_pipeline(ZarrsCodecPipeline)

__all__ = ["ZarrsCodecPipeline", "DiscontiguousArrayError", "CollapsedDimensionError"]
