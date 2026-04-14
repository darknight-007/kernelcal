"""kernelcal.urban — Spectral controller detection on urban building graphs.

Applies the MaxCal fixed-point kernel framework (P4) to OSM building
footprint proximity graphs.  Cities are treated as technosphere systems
where the controller is urban planning / zoning law.

Main entry: city_graph.py
"""
from .city_graph import fetch_buildings, buildings_to_graph, CityGraph
