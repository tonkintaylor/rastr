from rastr.raster import RasterModel

soft = RasterModel.example().taper_border(width=50, limit=1)
soft.plot()
