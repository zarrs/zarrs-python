from zarr.registry import register_pipeline

from .pipeline import ZarrsCodecPipeline as _ZarrsCodecPipeline


# Need to do this redirection so people can access the pipeline as `zarrs_python.ZarrsCodecPipeline` instead of `zarrs_python.pipeline.ZarrsCodecPipeline`
class ZarrsCodecPipeline(_ZarrsCodecPipeline):
    pass


register_pipeline(ZarrsCodecPipeline)

__all__ = ["ZarrsCodecPipeline"]
