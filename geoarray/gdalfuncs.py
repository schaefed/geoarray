#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gdal, osr
import numpy as np
import wrapper as ga
from gdalio import _getDataset, _fromDataset
from gdaltrans import _Projection, _Transformer

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

def _warpTo(source, target, max_error=0.125):

    if target.ndim == 1:
        target = target[None,:]
    if target.ndim < source.ndim:
        target = np.broadcast_to(
            target, source.shape[:-len(target.shape)]+target.shape, subok=True
        )
       
    target = np.ma.array(
        target,
        mask  = target==target.fill_value,
        dtype = source.dtype,
        copy  = True,
        subok = True
    )
    target[target.mask] = source.fill_value
    target.fill_value = source.fill_value

    out = _getDataset(target, True)
    resampling = gdal.GRA_NearestNeighbour
    
    gdal.ReprojectImage(
        _getDataset(source), out,
        None, None,
        resampling, 
        0.0, max_error
    )
    return _fromDataset(out)

def project(grid, proj, cellsize=None, max_error=0.125):

    bbox = grid.bbox
    proj = _Projection(proj)
    trans = _Transformer(grid._proj, proj)
    uly, ulx = trans(bbox["ymax"], bbox["xmin"])
    lry, lrx = trans(bbox["ymin"], bbox["xmax"])
    ury, urx = trans(bbox["ymax"], bbox["xmax"])
    lly, llx = trans(bbox["ymin"], bbox["xmin"])

    # Calculate cellsize, i.e. same number of cells along the diagonal.
    if not cellsize:
        src_diag = np.sqrt(grid.nrows**2 + grid.ncols**2)
        # trg_diag = np.sqrt((uly - lry)**2 + (lrx - ulx)**2)
        trg_diag = np.sqrt((lly - ury)**2 + (llx - urx)**2)
        cellsize = trg_diag/src_diag
    
    # number of cells
    ncols = int(abs(round((max(urx, lrx) - min(ulx, llx))/cellsize)))
    nrows = int(abs(round((max(ury, lry) - min(uly, lly))/cellsize)))
    
    target = ga.array(
        data       = np.full((grid.nbands, nrows, ncols), grid.fill_value, grid.dtype),
        fill_value = grid.fill_value,
        dtype      = grid.dtype,
        yorigin    = max(uly, ury, lly, lry),
        xorigin    = min(ulx, urx, llx, lrx),
        origin     = "ul",
        cellsize   = (-cellsize, cellsize),
        proj       = proj,
        mode       = grid.mode, 
    )

    return resample(
        source    = grid,
        target    = target,
        max_error = max_error,
    )   

def resample(source, target, max_error=0.125):
    return _warpTo(
        source=source,
        target=target,
        max_error=max_error
    )
    
