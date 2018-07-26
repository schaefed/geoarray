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
        elif value:
            if isinstance(value, int):
                method = self._srs.ImportFromProj4
                param = "+init=epsg:{:}".format(value)
            elif isinstance(value, dict):
                method = self._srs.ImportFromProj4
                param =  "+{:}".format(" +".join(
                    ["=".join(map(str, pp)) for pp in value.items()]))
            elif isinstance(value, str):
                method = self._srs.ImportFromWkt
                param = value
            else:
                raise RuntimeError("Projection not understood")

            if method(param):
                raise RuntimeError("Failed to set projection")

    def toWkt(self):
        return self._srs.ExportToPrettyWkt()

    def toProj4(self):
        out = self._srs.ExportToProj4()
        out = dict(filter(lambda x: len(x) == 2, [p.strip().split("=") for p in out.split("+")]))
        return out

    def __nonzero__(self):
        # is a an projection set?
        return bool(self.toWkt())

    def __get__(self, *args, **kwargs):
        return self

    def __set__(self, obj, val):
        self._import(val)

    def __str__(self):
        return self.toWkt()

    def __repr__(self):
        return self.toWkt()


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
            sproj._srs, tproj._srs)

    def __call__(self, y, x):
        try:
            xt, yt, _ = self._tx.TransformPoint(x, y)
        except NotImplementedError:
            raise AttributeError("Projections not correct or given!")
        return yt, xt
