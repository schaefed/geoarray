#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def _integerSlice(slc):
    return (slc, slc+1)

def _arraySlice(slc):
    if slc.dtype in (np.int32,np.int64,int):
        # integer array -> all values can be treated
        # as a single integer index, no matter what
        # shape the array has -> dimensions ??
        return [(slc.min(axis=d),slc.max(axis=d)+1) for d in xrange(slc.ndim)]
    elif slc.dtype == np.bool:
        # boolean array -> min(rows with true):max(rows with true)
        return [(min(d),max(d)) for d  in np.where(slc)]

def _sliceSlice(slc,shape):    
    return slc.indices(shape)[:2]

def slicingBounds(slc,shape):
    # processing
    out = []
    if not issubclass(type(slc),tuple):
        slc = (slc,)

    for dimslice,dimshape in zip(slc,shape):
        if issubclass(type(dimslice),int):
            out.append(_integerSlice(dimslice))
        elif issubclass(type(dimslice),np.ndarray):
            out.extend(_arraySlice(dimslice))
        elif hasattr(dimslice,"__iter__"):  # lists,tuple, generators
            out.extend(_arraySlice(np.array(dimslice)))
        elif issubclass(type(dimslice),slice):
            out.append(_sliceSlice(dimslice,dimshape))
        elif issubclass(type(dimslice),type(Ellipsis)):
            out.append(_sliceSlice(slice(None,None,None),dimshape))

    out.extend([_sliceSlice(slice(None,None,None),s) for s in shape[len(out):]])

    return out

    
