#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gdal, osr
import warnings

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

class _Geotrans(object):
    def __init__(self, yorigin, xorigin, ycellsize, xcellsize, yparam, xparam):
        self.yorigin = yorigin
        self.xorigin = xorigin
        self.ycellsize = ycellsize
        self.xcellsize = xcellsize
        self.yparam = yparam or 0
        self.xparam = xparam or 0

    def keys(self):
        return ("yorigin", "xorigin", "ycellsize", "xcellsize", "yparam", "xparam")

    def __getitem__(self, key):
        return getattr(self, key)

    def copy(self, **kwargs):
        args = dict(self)
        args.update(kwargs)
        return _Geotrans(**args)

    def __str__(self):
        return str({
            "yorigin": self.yorigin,
            "xorigin": self.xorigin,
            "ycellsize": self.ycellsize,
            "xcellsize": self.xcellsize,
            "yparam": self.yparam,
            "xparam": self.xparam})

    def toGdal(self):
        return (
            self.xorigin, self.xcellsize, self.xparam,
            self.yorigin, self.yparam, self.ycellsize)

    @classmethod
    def fromGdal(cls, geotrans):
        return _Geotrans(
            yorigin = geotrans[3],
            xorigin = geotrans[0],
            ycellsize = geotrans[5],
            xcellsize = geotrans[1],
            yparam = geotrans[4],
            xparam = geotrans[2])


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
