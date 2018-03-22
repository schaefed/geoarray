#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class GeotransMixin(object):
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

