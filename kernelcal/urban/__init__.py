"""kernelcal.urban — Spectral controller detection on urban building graphs.

Applies the MaxCal fixed-point kernel framework (P4) to OSM building
footprint proximity graphs.  Cities are treated as technosphere systems
where the controller is urban planning / zoning law.

Two graph-construction modes are supported:

* ``'knn'`` (historical default) — Euclidean k-NN on building centroids
  with Gaussian edge weights.
* ``'road_knn'`` (option 1) — k-NN on **road-network distance** between
  building centroids, using an OSM road graph snapped to each centroid.
  Produces a sparser, more zoning-shaped Laplacian than pure Euclidean
  proximity; useful for isolating the controller signature of grid
  boulevards vs organic street fabric.

Main entry: city_graph.py
"""
from .city_graph import (
    CityGraph,
    buildings_to_graph,
    buildings_to_graph_from_bbox,
    buildings_to_graph_via_roads,
    buildings_to_graph_via_roads_from_bbox,
    fetch_buildings,
    fetch_buildings_bbox,
    fetch_road_graph_bbox,
)

__all__ = [
    'CityGraph',
    'buildings_to_graph',
    'buildings_to_graph_from_bbox',
    'buildings_to_graph_via_roads',
    'buildings_to_graph_via_roads_from_bbox',
    'fetch_buildings',
    'fetch_buildings_bbox',
    'fetch_road_graph_bbox',
]
