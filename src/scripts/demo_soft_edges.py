from rastr.raster import RasterModel

soft = RasterModel.example().soft_edges(width=50, limit=1)
soft.plot()
