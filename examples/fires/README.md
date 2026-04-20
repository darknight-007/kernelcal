# examples/fires

Bobcat Fire vegetation segmentation and kernelcal integration.

| Script                       | What it does                                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| `bf_geotiff_export.py`       | Export polygon layers to GeoTIFF masks + multi-band composites.  Writes to `geotiffs/` at repo root. |
| `bf_kernelcal_demo.py`       | End-to-end kernelcal pipeline on BF temporal raster / vegetation masks.                              |
| `bf_kernelcal_plots.py`      | Re-plot saved kernelcal outputs for the BF paper.                                                    |
| `bf_spatial_overlay.py`      | Spatial-overlay figures merging BF detections with basemap tiles.                                    |
| `bf_vegetation_segment.py`   | Batch Grounded-SAM-2 vegetation segmentation.  Requires a reachable GPU server —                     |
|                              | configure via `KERNELCAL_GROUNDED_SAM_URL` / `KERNELCAL_GROUNDING_DINO_URL`.                         |

Dataset: MBTiles + polygon data under the repo-root `datasets/bf_mbtiles/`
and `datasets/` directories.
