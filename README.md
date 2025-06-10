# Introduction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15631927.svg)](https://doi.org/10.5281/zenodo.15631927)

The gravity data used here is extracted from Budiman (1991) thesis "[Interpretation of gravity data over central Jawa [_sic_], Indonesia](https://librarysearch.adelaide.edu.au/permalink/61ADELAIDE_INST/rinku3/alma992913301811)". This data consists of 2811 complete Bouguer anomaly data collected during ground surveys in Central Java between 1965 to 1977, reduced using a Bouguer and terrain density of 2670 $\frac{kg}{m^3}$ from the observed gravity acceleration values. Gravity data X (Easting, in metres) and Y (Northing, in metres) positions are originally given in the local ["Batavia TM 109 SE"](https://epsg.io/2308) coordinate system, whose .wkt file is included here and read using `pyproj.CRS.from_wkt` function. Gravity data elevation is sampled from [BATNAS'](https://sibatnas.big.go.id/) 200 m resolution DEM and bathymetry (using QGIS' `Raster analysis`>`Sample raster values`), so they are relative to the mean local sea level/geoid.

We cannot locate any explanation of this gravity data positioning technique in the literature, so the gravity anomaly uncertainty cannot be well estimated. Due to the absence of a reliable uncertainty estimate, the central Java gravity anomaly data presented here is _not good_ for scientific publication but is good enough for experimenting with the equivalent surface gridding technique as described in [Dampney (1969)](https://doi.org/10.1190/1.1439996).

So let's start!
