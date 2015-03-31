#! /usr/bin/env python
# -*- coding: utf-8 -*-

from .src.geogrid import GeoGrid
from .src.geogridfuncs import (
    indexCoordinates,
    coordinateIndex, 
    addCells,     
    removeCells,
    enlargeGrid,
    shrinkGrid,
    mergeGrid,
    maskGrid,
    trimGrid,
    snapGrid
)
    

__all__ = ["GeoGrid", shrinkGrid]

# Delete src from package namespace -> no import geogrid.src 
del src

