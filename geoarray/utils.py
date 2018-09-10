#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


try:
    basestring
except NameError:
    basestring = str


def _broadcastTo(array, shape, dims):
    """
    array, shape: see numpy.broadcast_to
    dims: tuple, the dimensions the array dimensions
          should end up in the output array
    """
    assert len(array.shape) == len(dims)
    assert len(set(dims)) == len(dims)  # no duplicates
    # handle negative indices
    dims = [d if d >= 0 else d+len(shape) for d in dims]
    # bring array to the desired dimensionality
    slc = [slice(None, None, None) if i in dims else None
           for i in range(len(shape))]
    return np.broadcast_to(array[tuple(slc)], shape)


def _broadcastedMeshgrid(*arrays):

    def _toNd(array, n, pos=-1):
        """
        expand given 1D array to n dimensions,
        the dimensions > 0 can be given by pos
        """
        assert array.ndim == 1, "arrays should be 1D"
        shape = np.ones(n, dtype=int)
        shape[pos] = len(array)
        return arr.reshape(shape)

    shape = tuple(len(arr) for arr in arrays)

    out = []
    for i, arr in enumerate(arrays):
        tmp = np.broadcast_to(
            _toNd(arr, len(shape), pos=i),
            shape
        )
        # there should be a solution without transposing...
        out.append(tmp.T)
    return out


def _tupelize(arg):
    out = arg
    if isinstance(out, basestring):
        out = (out, )
    if not isinstance(out, tuple):
        try:
            out = tuple(out)
        except TypeError:
            out = (out, )
    return out
