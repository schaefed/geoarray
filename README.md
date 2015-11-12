#geoarray

#Purpose
This module provides a numpy.ma.MaskedArray subclass and a number of wrapper 
functions to facilitate the work with array-like data in a geographically explicit 
context. The Python GDAL bindings are used for I/O.

#Requirements
- GDAL >= 1.11
- numpy >= 1.8

#General
This module tries to imitate the general numpy functionality as closly as possible.
As a MaskedArray subclass a GeoArray Instance is (hopefully) usable wherever its parents are.

On Linux the GDAL virtual memory mapping is used, i.e. data is only read from disk, if it is
actually accessed. This feature saves a lot of initialization time and memory, but is not available
for other OS.

#Usage
There are a bunch of wrapper functions like ```array, zeros, ones, empty, full``` and their
```*_like``` companions. They do what what their numpy counterparts do.
The probably most import factory function might however be ```fromfile```.

A basic example:

```python

# import the module to an handy alias
import geoarray as ga

# read the dataset
grid = ga.fromfile("yourfile.tif")

# do some work
grid += np.exp(grid)
print(grid.sum())

# get a slice
rect = grid[::2,5:10]

# see how this affects the metadata
print(rect.getOrigin())
print(rect.shape)
print(rect.bbox)

# write the new dataset to disk
rect.tofile("test.asc")
```
For more examples see the doctests in geoarray.py

#Restrictions
Although GeoArray has an attribute proj_params, which might hold projection parameters as a proj4
compatible dictionary, this information is currently unused. The only restriction when doing
calculations with GeoArray instances are those of numpy: if two objects have the same shape or are broadcastable,
these computations will be done, regardless of mismatching map projections. 
