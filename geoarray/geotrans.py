#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .utils import _broadcastTo
from abc import ABCMeta, abstractmethod, abstractproperty

class _GeoBase(object):
    __metaclass__ = ABCMeta

    #@property
    #def origin(self):
    #    return "".join(
    #        ["l" if self.ycellsize > 0 else "u",
    #         "l" if self.xcellsize > 0 else "r"])

    @abstractproperty
    def coordinates(self):
        pass

    @abstractproperty
    def bbox(self):
        pass

    @abstractmethod
    def _todict(self):
        pass

    @abstractmethod
    def _replace(self):
        pass

    @abstractmethod
    def _getitem(self):
        pass

    @abstractmethod
    def toGdal(self):
        raise NotImplementedError


class _Geolocation(_GeoBase):
    def __init__(self, yvalues, xvalues, shape, origin):
        self.yvalues = yvalues
        self.xvalues = xvalues
        self.shape = shape
        self.origin = origin

    @property
    def yorigin(self):
        bbox = self.bbox
        return self.bbox["ymax" if self.origin[0] == "u" else "ymin"]

    @property
    def xorigin(self):
        return self.bbox["xmax" if self.origin[1] == "r" else "xmin"]

    @property
    def coordinates(self):
        return self.yvalues, self.xvalues

    @property
    def bbox(self):

        ymin, ymax = self.yvalues.min(), self.yvalues.max()
        xmin, xmax = self.xvalues.min(), self.xvalues.max()

        ydiff = np.abs(np.diff(self.yvalues, axis=-2))
        xdiff = np.abs(np.diff(self.xvalues, axis=-1))
        if self.origin[0] == "u":
            ymin -= ydiff[-1].max()
        else:
            ymax += ydiff[0].max()

        if self.origin[1] == "l":
            xmax += xdiff[:,-1].max()
        else:
            xmin -= xdiff[:,0].max()

        return {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax}


    def _replace(self, yvalues=None, xvalues=None, origin=None, shape=None):
        return _Geolocation(
            yvalues=self.yvalues if yvalues is None else yvalues,
            xvalues=self.xvalues if xvalues is None else xvalues,
            shape=self.shape if shape is None else shape,
            origin=self.origin if origin is None else origin)

    def _todict(self):
        return {
            "yvalues": self.yvalues,
            "xvalues": self.xvalues}

    def _getitem(self, slc):

        yvalues = np.array(
            _broadcastTo(self.yvalues, self.shape, (-2, -1))[slc],
            copy=False, ndmin=2)

        xvalues = np.array(
            _broadcastTo(self.xvalues, self.shape, (-2, -1))[slc],
            copy=False, ndmin=2)

        if yvalues.ndim > 2:
            yvalues = yvalues[..., 0, :, :]

        if xvalues.ndim > 2:
            xvalues = xvalues[..., 0, :, :]

        return self._replace(yvalues=yvalues, xvalues=xvalues)

    def toGdal(self):
        return {
            "X_BAND": self.shape[0],  # need to be filled
            "Y_BAND": self.shape[0] + 1,  # need to be filled
            "PIXEL_OFFSET": 0,
            "LINE_OFFSET": 0,
            "PIXEL_STEP": 1,
            "LINE_STEP": 1}


class _Geotrans(_GeoBase):
    def __init__(self, yorigin, xorigin, ycellsize, xcellsize,
                 yparam, xparam, origin, shape):
        self.yorigin = yorigin
        self.xorigin = xorigin
        self.ycellsize = ycellsize
        self.xcellsize = xcellsize
        self.yparam = yparam
        self.xparam = xparam
        self.origin = origin
        self.shape = shape
        self._yvalues = None
        self._xvalues = None

    @property
    def cellsize(self):
        return (self.ycellsize, self.xcellsize)

    @property
    def nrows(self):
        try:
            return self.shape[-2]
        except IndexError:
            return 1

    @property
    def ncols(self):
        try:
            return self.shape[-1]
        except IndexError:
            return 1

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
        out = (self.xorigin, self.xcellsize, self.xparam,
                self.yorigin, self.yparam, self.ycellsize)
        return out

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
                 yparam=None, xparam=None, origin=None, shape=None):

        return _Geotrans(
            yorigin=self.yorigin if yorigin is None else yorigin,
            xorigin=self.xorigin if xorigin is None else xorigin,
            ycellsize=self.ycellsize if ycellsize is None else ycellsize,
            xcellsize=self.xcellsize if xcellsize is None else xcellsize,
            yparam=self.yparam if yparam is None else yparam,
            xparam=self.xparam if xparam is None else xparam,
            origin=self.origin if origin is None else origin,
            shape=self.shape if shape is None else shape)

    def _todict(self):
        return {
            "yorigin": self.yorigin,
            "xorigin": self.xorigin,
            "ycellsize": self.ycellsize,
            "xcellsize": self.xcellsize,
            "yparam": self.yparam,
            "xparam": self.xparam,
            "origin": self.origin}

    def _getitem(self, slc):

        yvalues = np.array(
            _broadcastTo(self.yvalues, self.shape, (-2, -1))[slc],
            copy=False, ndmin=2)

        xvalues = np.array(
            _broadcastTo(self.xvalues, self.shape, (-2, -1))[slc],
            copy=False, ndmin=2)

        nrows, ncols = yvalues.shape[-2:]
        ycellsize = np.diff(yvalues, axis=-2).mean() if nrows > 1 else self.ycellsize
        xcellsize = np.diff(xvalues, axis=-1).mean() if ncols > 1 else self.xcellsize

        out = self._replace(
            yorigin=yvalues.max(), xorigin=xvalues.min(),
            ycellsize=ycellsize, xcellsize=xcellsize,
            shape=yvalues.shape)
        return out
