#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .utils import _broadcastTo


class _Geotrans(object):
    def __init__(self, yorigin, xorigin, ycellsize, xcellsize,
                 yparam, xparam, nrows, ncols):
        self.yorigin = yorigin
        self.xorigin = xorigin
        self.ycellsize = ycellsize
        self.xcellsize = xcellsize
        self.yparam = yparam
        self.xparam = xparam
        self.nrows = nrows
        self.ncols = ncols
        self._yvalues = None
        self._xvalues = None

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
        if self._yvalues is None or self._xvalues is None:
            xdata, ydata = np.meshgrid(
                np.arange(self.ncols, dtype=float),
                np.arange(self.nrows, dtype=float))
            self._yvalues, self._xvalues = self._calcCoordinate(ydata, xdata)
        return self._yvalues, self._xvalues

    @property
    def yvalues(self):
        return self.coordinates[0]

    @property
    def xvalues(self):
        return self.coordinates[1]

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

    def _replace(self, yorigin=None, xorigin=None, ycellsize=None, xcellsize=None,
                 yparam=None, xparam=None, nrows=None, ncols=None):

        return _Geotrans(
            yorigin=self.yorigin if yorigin is None else yorigin,
            xorigin=self.xorigin if xorigin is None else xorigin,
            ycellsize=self.ycellsize if ycellsize is None else ycellsize,
            xcellsize=self.xcellsize if xcellsize is None else xcellsize,
            yparam=self.yparam if yparam is None else yparam,
            xparam=self.xparam if xparam is None else xparam,
            nrows=self.nrows if nrows is None else nrows,
            ncols=self.ncols if ncols is None else ncols)

    def _todict(self):
        return {
            "yorigin": self.yorigin,
            "xorigin": self.xorigin,
            "ycellsize": self.ycellsize,
            "xcellsize": self.xcellsize,
            "yparam": self.yparam,
            "xparam": self.xparam}

    def _getitem(self, shape, slc):

        yvalues = np.array(
            _broadcastTo(self.yvalues, shape, (-2, -1))[slc],
            copy=False, ndmin=2)

        xvalues = np.array(
            _broadcastTo(self.xvalues, shape, (-2, -1))[slc],
            copy=False, ndmin=2)

        nrows, ncols = yvalues.shape[-2:]
        ycellsize = np.diff(yvalues, axis=-2).mean() if nrows > 1 else self.ycellsize
        xcellsize = np.diff(xvalues, axis=-1).mean() if ncols > 1 else self.xcellsize

        out = _Geotrans(
            yparam=self.yparam, xparam=self.xparam,
            yorigin=yvalues.max(), xorigin=xvalues.min(),
            ycellsize=ycellsize, xcellsize=xcellsize,
            nrows=nrows, ncols=ncols)
        return out


