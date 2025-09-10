from rastr.create import raster_from_point_cloud

x = [10, 1, 1.5, 3, 30, 5]
y = [0, 1, 1.5, 5, 7, 9]
raster = raster_from_point_cloud(
    x=x,
    y=y,
    z=[10, 20, 30, 40, 50, 60],
    crs="EPSG:32633",
)
ax = raster.plot()
ax.scatter(x, y, c="red")
