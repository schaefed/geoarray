#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import namedtuple
from .utils import _broadcastTo

_Geotrans = namedtuple("_Geotrans",
                      ("yorigin", "xorigin", "ycellsize", "xcellsize",
                       "yparam", "xparam", "geoloc"))


class GeotransMixin(object):
    @property
    def origin(self):
        return "".join(
            ["l" if self.geotrans.ycellsize > 0 else "u",
             "l" if self.geotrans.xcellsize > 0 else "r"])

    @property
    def bbox(self):

        corners = np.array(self.getCorners())
        ymin, xmin = np.min(corners, axis=0)
        ymax, xmax = np.max(corners, axis=0)

        return {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax}


    def _calcCoordinate(self, row, col):

        yval = (self.geotrans.yorigin
                + col * self.geotrans.yparam
                + row * self.geotrans.ycellsize)
        xval = (self.geotrans.xorigin
                + col * self.geotrans.xcellsize
                + row * self.geotrans.xparam)
        return yval, xval

    def toGdal(self):
        return (self.geotrans.xorigin, self.geotrans.xcellsize, self.geotrans.xparam,
                self.geotrans.yorigin, self.geotrans.yparam, self.geotrans.ycellsize)

    @property
    def cellsize(self):
        return (self.geotrans.ycellsize, self.geotrans.xcellsize)

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
        if self._yvalues is None:
            return self.coordinates[0]
        return self._yvalues

    @property
    def xvalues(self):
        if self._xvalues is None:
            return self.coordinates[1]
        return self._xvalues

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

    def __getitem__(self, slc):

        yvalues = np.array(
            _broadcastTo(self.yvalues, self.shape, (-2, -1))[slc],
            copy=False, ndmin=2)

        xvalues = np.array(
            _broadcastTo(self.xvalues, self.shape, (-2, -1))[slc],
            copy=False, ndmin=2)

        # if self.geoloc is not False:
        #     raise NotImplementedError

        nrows, ncols = yvalues.shape[-2:]
        ycellsize = np.diff(yvalues, axis=-2).mean() if nrows > 1 else self.ycellsize
        xcellsize = np.diff(xvalues, axis=-1).mean() if ncols > 1 else self.xcellsize

        return self.geotrans._replace(
            yorigin=yvalues.max(), xorigin=xvalues.min(),
            ycellsize=ycellsize, xcellsize=xcellsize)


