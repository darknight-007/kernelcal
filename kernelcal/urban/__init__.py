"""kernelcal.urban — Spectral controller detection on urban building graphs.

Applies the MaxCal fixed-point kernel framework (P4) to OSM building
footprint proximity graphs.  Cities are treated as technosphere systems
where the controller is urban planning / zoning law.

Main entry: city_graph.py
"""
from .city_graph import (
    CityGraph,
    buildings_to_graph,
    buildings_to_graph_from_bbox,
    fetch_buildings,
    fetch_buildings_bbox,
)

__all__ = [
    'CityGraph',
    'buildings_to_graph',
    'buildings_to_graph_from_bbox',
    'fetch_buildings',
    'fetch_buildings_bbox',
]
