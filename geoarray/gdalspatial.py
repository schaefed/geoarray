#! /usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import gdal, osr

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')


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
