#Slicing Semantics
There are basically three possibly ways to handle slicing in the provided georgaphic context:
1. The index corners (i.e. the first row/cloumn, last row/coloumn) mark the selected rectangle and the cellsize is adjusted accordingly. That implies, that
   the cellsize of an GeoArray instance will change from 500 to 1000 when indexed with Slice(None, None, 2).
   Things get more complicated with fancy slicing, though. How to deal with indices like [0,5,1]? This destroys the lower than relation along the coordinate axis.
   Following the same the semantics as above would reduce the cellsize in the given example to 250.
2. Only the upper left index corner marks the geographic selection. The cellsize is fixed and the lower/right grid bound is calculated as the product of
   cellsize and nrows/ncols. This seems to be more sensible with fancy slicing but wrong for indexes like Slice(3, 10, None). Things get complicated with
   grid origin unequal to the upper left corner.
3. A combination of both. Scheme 1 for normal, scheme 2 for fancy slicing. In a theoretical point of view this makes most sense to me, but would introduce an
   rather strong semantic inconsistency within one operation (i.e. slicing an GeoArray instance) and will lead to a more complicated and most likely ugly
   implementation.
   
