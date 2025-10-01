import json
import sys

import zarr

# imported_modules must be determined first
imported_modules = list(sys.modules)
with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}):
    try:
        zarr.array([])
    except zarr.core.config.BadConfigError:
        is_registered = False
    else:
        is_registered = True

print(
    json.dumps(
        dict(
            imported_modules=imported_modules,
            is_registered=is_registered,
        )
    )
)
