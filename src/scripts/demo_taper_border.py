from rastr.raster import Raster

soft = Raster.example().taper_border(width=50, limit=1)
soft.plot()
