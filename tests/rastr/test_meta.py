import numpy as np
import pytest
from affine import Affine
from pydantic import ValidationError
from pyproj.crs.crs import CRS

from rastr.raster import RasterMeta, RasterModel

_NZTM_CRS = CRS.from_epsg(2193)


class TestRaster:
    def test_instantiable(self):
        """Test that Raster can be instantiated."""
        arr = np.array([[1, 2], [3, 4]])
        meta = RasterMeta(cell_size=1, crs=_NZTM_CRS, transform=Affine.scale(1.0, 1.0))
        raster = RasterModel(arr=arr, raster_meta=meta)
        assert isinstance(raster, RasterModel)

    def test_3d_fails(self):
        """Test that Raster cannot be instantiated with a 3D array."""
        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])  # 2 x 2 x 2 array
        meta = RasterMeta(cell_size=1, crs=_NZTM_CRS, transform=Affine.scale(1.0, 1.0))

        with pytest.raises(ValidationError):
            RasterModel(arr=arr, raster_meta=meta)

    def test_2x2x1_fails(self):
        arr = np.array([[[0.23465047], [0.77642868]], [[0.92393235], [0.55804058]]])
        meta = RasterMeta(cell_size=1, crs=_NZTM_CRS, transform=Affine.scale(1.0, 1.0))
        with pytest.raises(ValidationError):
            RasterModel(arr=arr, raster_meta=meta)

    class TestExample:
        def test_example(self):
            # Arrange
            arr = np.array([[1, 2], [3, 4]])
            meta = RasterMeta(
                cell_size=1,
                crs=_NZTM_CRS,
                transform=Affine.scale(1.0, 1.0),
            )

            # Act
            example_meta = RasterModel(
                arr=arr,
                raster_meta=meta,
            )

            # Assert
            assert isinstance(example_meta, RasterModel)

    def test_get_cell_centre_coords(self):
        # Arrange
        cell_size = 2.0
        crs = _NZTM_CRS
        transform = Affine.translation(100, 200) * Affine.scale(cell_size, -cell_size)
        meta = RasterMeta(cell_size=cell_size, crs=crs, transform=transform)
        shape = (2, 3)

        # Act
        coords = meta.get_cell_centre_coords(shape)

        # Assert
        assert coords.shape == (2, 3, 2)
        expected = np.array(
            [
                [[101.0, 199.0], [103.0, 199.0], [105.0, 199.0]],
                [[101.0, 197.0], [103.0, 197.0], [105.0, 197.0]],
            ]
        )
        np.testing.assert_allclose(coords, expected)
