#! /usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import gdal, osr

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')


class _Geotrans(object):
    def __init__(
            self, yorigin, xorigin, ycellsize, xcellsize, yparam, xparam, *args, **kwargs):
        self.yorigin = yorigin
        self.xorigin = xorigin
        self.ycellsize = ycellsize
        self.xcellsize = xcellsize
        self.yparam = yparam
        self.xparam = xparam

    @property
    def origin(self):
        return "".join(
            ["l" if self.ycellsize > 0 else "u",
             "l" if self.xcellsize > 0 else "r"])

    @property
    def bbox(self):

        corners = np.array(self.getCorners())
        ymin, xmin = np.min(corners, axis=0)
        ymax, xmax = np.max(corners, axis=0)

        return {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax}


    def _calcCoordinate(self, row, col):
        yval = (self.yorigin
                + col * self.yparam
                + row * self.ycellsize)
        xval = (self.xorigin
                + col * self.xcellsize
                + row * self.xparam)
        return yval, xval

    def toGdal(self):
        return (self.xorigin, self.xcellsize, self.xparam,
                self.yorigin, self.yparam, self.ycellsize)

    @property
    def cellsize(self):
        return (self.ycellsize, self.xcellsize)

    @property
    def coordinates(self):
        # NOTE: rather costly, should be cached
        xdata, ydata = np.meshgrid(
            np.arange(self.ncols, dtype=float),
            np.arange(self.nrows, dtype=float))
        return self._calcCoordinate(ydata, xdata)

    def getCorners(self):
        corners = [(0, 0), (self.nrows, 0),
                   (0, self.ncols), (self.nrows, self.ncols)]
        return [self._calcCoordinate(*idx) for idx in corners]

    def getCorner(self, corner=None):
        if not corner:
            corner = self.origin

        bbox = self.bbox
        return (
            bbox["ymax"] if corner[0] == "u" else bbox["ymin"],
            bbox["xmax"] if corner[1] == "r" else bbox["xmin"],)


class _Projection(object):
    def __init__(self, arg):
        """
        Arguments:
        arg can be:
        1. int  : EPSG code
        2. dict : pyproj compatable dictionary
        3. str  : WKT string
        4. _Projection
        """
        self._srs = osr.SpatialReference()
        self._import(arg)

    def _import(self, value):
        if isinstance(value, _Projection):
            self._srs = value._srs
        elif isinstance(value, int):
            self._srs.ImportFromProj4("+init=epsg:{:}".format(value))
        elif isinstance(value, dict):
            params =  "+{:}".format(" +".join(
                ["=".join(map(str, pp)) for pp in value.items()]))
            self._srs.ImportFromProj4(params)
        elif isinstance(value, str):
            self._srs.ImportFromWkt(value)

        if value and self is None:
            warnings.warn("Projection not understood", RuntimeWarning)

    def _export(self):
        out = self._srs.ExportToPrettyWkt()
        return out or None

    def __nonzero__(self):
        # is a an projection set?
        return self._export() is not None

    def __get__(self, obj, objtype):
        return self._export()

    def __set__(self, obj, val):
        self._import(val)


class _Transformer(object):
    def __init__(self, sproj, tproj):
        """
        Arguments
        ---------
        sproj, tproj : Projection

        Purpose
        -------
        Encapsulates the osr Cordinate Transformation functionality
        """
        self._tx = osr.CoordinateTransformation(
            sproj._srs, tproj._srs
        )

    def __call__(self, y, x):
        try:
            xt, yt, _ = self._tx.TransformPoint(x, y)
        except NotImplementedError:
            raise AttributeError("Projections not correct or given!")
        return yt, xt
