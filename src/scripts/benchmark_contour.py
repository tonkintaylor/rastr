import numpy as np
import pyinstrument

from rastr.raster import Raster

raster = Raster.example()

with pyinstrument.profile():
    contour_gdf = raster.contour(np.linspace(-1, 1, 1000))
