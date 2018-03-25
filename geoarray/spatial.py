#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import floor, ceil


class SpatialMixin(object):

    def trim(self):
        """
        Arguments
        ---------
        None

        Returns
        -------
        GeoArray

        Purpose
        -------
        Removes rows and columns from the margins of the
        grid if they contain only fill values.
        """

        try:
            y_idx, x_idx = np.where(self.data != self.fill_value)
            return self.removeCells(
                top=min(y_idx), bottom=self.nrows - max(y_idx) - 1,
                left=min(x_idx), right=self.ncols - max(x_idx) - 1)
        except ValueError:
            return self

    def removeCells(self, top=0, left=0, bottom=0, right=0):
        """
        Arguments
        ---------
        top, left, bottom, right : int

        Returns
        -------
        GeoArray

        Purpose
        -------
        Remove the number of given cells from the respective
        margin of the grid.
        """

        top = int(max(top, 0))
        left = int(max(left, 0))
        bottom = self.nrows - int(max(bottom, 0))
        right = self.ncols - int(max(right, 0))

        return self[..., top:bottom, left:right]

    def shrink(self, ymin=None, ymax=None, xmin=None, xmax=None):
        """
        Arguments
        ---------
        ymin, ymax, xmin, xmax : scalar

        Returns
        -------
        GeoArray

        Purpose
        -------
        Shrinks the grid in a way that the given bbox is still
        within the grid domain.

        BUG:
        ------------
        For bbox with both negative and postive values
        """
        bbox = {
            "ymin": ymin if ymin is not None else self.bbox["ymin"],
            "ymax": ymax if ymax is not None else self.bbox["ymax"],
            "xmin": xmin if xmin is not None else self.bbox["xmin"],
            "xmax": xmax if xmax is not None else self.bbox["xmax"],
            }

        cellsize = [float(abs(cs)) for cs in self.cellsize]
        top = floor((self.bbox["ymax"] - bbox["ymax"]) / cellsize[0])
        left = floor((bbox["xmin"] - self.bbox["xmin"]) / cellsize[1])
        bottom = floor((bbox["ymin"] - self.bbox["ymin"]) / cellsize[0])
        right = floor((self.bbox["xmax"] - bbox["xmax"]) / cellsize[1])

        return self.removeCells(
            max(top, 0), max(left, 0), max(bottom, 0), max(right, 0))

    def addCells(self, top=0, left=0, bottom=0, right=0):
        """
        Arguments
        ---------
        top, left, bottom, right : int

        Returns
        -------
        GeoArray

        Purpose
        -------
        Add the number of given cells to the respective margin of the grid.
        """

        from .core import GeoArray

        top = int(max(top, 0))
        left = int(max(left, 0))
        bottom = int(max(bottom, 0))
        right = int(max(right, 0))

        if self.origin[0] == "l":
            top, bottom = bottom, top
        if self.origin[1] == "r":
            left, right = right, left

        shape = list(self.shape)
        shape[-2:] = self.nrows + top + bottom, self.ncols + left + right

        try:
            data = np.full(shape, self.fill_value, self.dtype)
        except TypeError:
            # fill_value is set to none
            raise AttributeError(
                "Valid fill_value needed, actual value is {:}"
                .format(self.fill_value))

        yorigin = self.yorigin + top*self.ycellsize * -1
        xorigin = self.xorigin + left*self.xcellsize * -1

        out = GeoArray(
            **self._getArgs(
                data=data,
                geotrans=self.geotrans._replace(
                    yorigin=yorigin, xorigin=xorigin, nrows=shape[-2], ncols=shape[-1]),
                mode="r", fobj=None))

        # the Ellipsis ensures that the function works
        # for arrays with more than two dimensions
        out[..., top:top+self.nrows, left:left+self.ncols] = self
        return out

    def enlarge(self, ymin=None, ymax=None, xmin=None, xmax=None):
        """
        Arguments
        ---------
        ymin, ymax, xmin, xmax : scalar

        Returns
        -------
        None

        Purpose
        -------
        Enlarge the grid in a way that the given coordinates will
        be part of the grid domain. Added rows/cols are filled with
        the grid's fill value.
        """

        bbox = {
            "ymin": ymin if ymin is not None else self.bbox["ymin"],
            "ymax": ymax if ymax is not None else self.bbox["ymax"],
            "xmin": xmin if xmin is not None else self.bbox["xmin"],
            "xmax": xmax if xmax is not None else self.bbox["xmax"],}

        cellsize = [float(abs(cs)) for cs in self.cellsize]

        top = ceil((bbox["ymax"] - self.bbox["ymax"]) / cellsize[0])
        left = ceil((self.bbox["xmin"] - bbox["xmin"]) / cellsize[1])
        bottom = ceil((self.bbox["ymin"] - bbox["ymin"]) / cellsize[0])
        right = ceil((bbox["xmax"] - self.bbox["xmax"]) / cellsize[1])

        return self.addCells(
            max(top, 0), max(left, 0), max(bottom, 0), max(right, 0))

    def coordinatesOf(self, y_idx, x_idx):
        """
        Arguments
        ---------
        y_idx, x_idx :  int

        Returns
        -------
        (scalar, scalar)

        Purpose
        -------
        Return the coordinates of the grid cell definied by the given
        row and column index values. The cell corner to which the returned
        values belong is definied
        by the attribute origin:
            "ll": lower-left corner
            "lr": lower-right corner
            "ul": upper-left corner
            "ur": upper-right corner
        """

        if ((y_idx < 0 or x_idx < 0)
            or (y_idx >= self.nrows
                or x_idx >= self.ncols)):
            raise ValueError("Index out of bounds !")

        yorigin, xorigin = self.getCorner("ul")
        return (
            yorigin - y_idx * abs(self.cellsize[0]),
            xorigin + x_idx * abs(self.cellsize[1]))

    def indexOf(self, ycoor, xcoor):
        """
        Arguments
        ---------
        ycoor, xcoor : scalar

        Returns
        -------
        (int, int)

        Purpose
        -------
        Find the grid cell into which the given coordinates
        fall and return its row/column index values.
        """


        yorigin, xorigin = self.getCorner("ul")

        yidx = int(floor((yorigin - ycoor) / float(abs(self.ycellsize))))
        xidx = int(floor((xcoor - xorigin) / float(abs(self.xcellsize))))

        if yidx < 0 or yidx >= self.nrows or xidx < 0 or xidx >= self.ncols:
            raise ValueError("Given Coordinates not within the grid domain!")

        return yidx, xidx


    # def snap(self,target):
    #     """
    #     Arguments
    #     ---------
    #     target : GeoArray

    #     Returns
    #     -------
    #     None

    #     Purpose
    #     -------
    #     Shift the grid origin that it matches the nearest cell origin in target.

    #     Restrictions
    #     ------------
    #     The shift will only alter the grid coordinates. No changes to the
    #     data will be done. In case of large shifts the physical integrety
    #     of the data might be disturbed!

    #     diff = np.array(self.getCorner()) - np.array(target.getCorner(self.origin))
    #     dy, dx = abs(diff)%target.cellsize * np.sign(diff)

    #     if abs(dy) > self.cellsize[0]/2.:
    #         dy += self.cellsize[0]

    #     if abs(dx) > self.cellsize[1]/2.:
    #         dx += self.cellsize[1]

    #     self.xorigin -= dx
    #     self.yorigin -= dy
