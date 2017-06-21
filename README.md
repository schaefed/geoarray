# geoarray

# Purpose
This python GDAL wrapper module provides a numpy.ma.MaskedArray subclass and a number of initializer 
functions to facilitate the work with array-like data in a geographically explicit context.

# Requirements
- GDAL >= 1.11
- numpy >= 1.11

# General
This module tries to imitate the general numpy functionality as closly as possible.
As a MaskedArray subclass a GeoArray Instance is (hopefully) usable wherever its parents are.

# Usage

## I/O

Existing files can be read with the ```fromfile``` function.

```python
# read the dataset
grid = ga.fromfile("yourfile.tif")
```

There are a bunch of wrapper functions like ```array, zeros, ones, empty, full``` which do
what their numpy counterparts do.

```python
# import the module to an handy alias
import geoarray as ga

# the most basic initialization gives not much more than a MaskedArray
grid = ga.zeros((300,250))

# Add some geospatial information
grid = ga.zeros((300,250), yorigin=1000, xorigin=850, cellsize=50)

# The origin of the grid defaults to the upper left corner.
# The options are "ul", "ll", "ur", "lr", i. e. "upper left", "lower left", "upper right", "lower right" 
grid = ga.zeros((300,250), yorigin=1000, xorigin=850, cellsize=50, origin="ll")

# If no fill_value is given, the smallest value of the datatype is used
grid = ga.zeros((300,250), yorigin=1000, xorigin=850, cellsize=50, origin="ll", fill_value=-9999)

# You can add projection information as a pyproj compatible dictionary, a wkt string or epsg code
grid = ga.zeros((300,250), yorigin=1000, xorigin=850, cellsize=50, origin="ll", fill_value=-9999, proj=3857)

```

## Arithmetic

As a subclass of MaskedArray (and therefore also of ndarray) GeoArray instances can be passed to
all numpy functions and accept the usual operators

```python
grid *= .8

grid2 = grid + 42

grid3 = np.exp(grid)
```

## Transformations

Coordinate transformations are as easy as
```python
pgrid = ga.project(grid, proj=2062)
```

# Slicing
GeoArray overrides the usual slicing behaviour in order to preserve the spatial context. The yorigin 
and xorigin attributes are updated according to the given origin of the instance.

The geographic extend of the array is determined by the first/last given index: 

```python
# grid with ymax=15000 and ymin=12000
grid = ga.zeros((300,250), origin="ul", cellsize=10, yorigin=15000, xorigin=10000)

# grid with ymax=14980 and ymin=12020
grid[2:-2]
```

Skipping cells along the coordinate axis, will alter the cellsize:
```python
# grid with ymax=14980, ymin=12020 and cellsize=(-20,10)
grid[2:-2:2]
```

Fancy slicing follows the same semantics:
```python
# grid with ymax=15000, ymin=14720 and cellsize=(-70,10)
grid[np.array([0,3,6,21])]
```

Note, that the *last* index limits the rectangle, *not* the highest:
```python
# grid with ymax=15000, ymin=14920 and cellsize=(-20,10)
grid[np.array([0,3,21,6])]
```

# Restrictions
- GDAL supports many different raster data formats, but only the Geotiff, Arc/Info Ascii Grid, Erdas Imagine and PNG formats are currently supported by geoarray.
- When converting between data formats, GDAL automatically adjusts the datatypes and truncates values. You might loose information that way.
