"""Tests for dtype preservation across various raster operations."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pyproj.crs.crs import CRS

from rastr.meta import RasterMeta
from rastr.raster import Raster


class TestDtypePreservation:
    """Test suite for dtype preservation across raster operations."""

    @pytest.fixture
    def float32_raster(self) -> Raster:
        """Create a float32 raster for testing."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        return Raster(arr=arr, raster_meta=meta)

    @pytest.fixture
    def float64_raster(self) -> Raster:
        """Create a float64 raster for testing."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        return Raster(arr=arr, raster_meta=meta)

    @pytest.fixture
    def float16_raster(self) -> Raster:
        """Create a float16 raster for testing (e.g., from integer conversion)."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        return Raster(arr=arr, raster_meta=meta)

    def test_addition_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that addition with scalar preserves float32."""
        result = float32_raster + 1.0
        assert result.arr.dtype == np.float32

    def test_addition_preserves_dtype_float64(self, float64_raster: Raster):
        """Test that addition with scalar preserves float64."""
        result = float64_raster + 1.0
        assert result.arr.dtype == np.float64

    def test_addition_preserves_dtype_float16(self, float16_raster: Raster):
        """Test that addition with scalar preserves float16."""
        result = float16_raster + 1.0
        assert result.arr.dtype == np.float16

    def test_raster_addition_preserves_dtype(self, float32_raster: Raster):
        """Test that raster-to-raster addition preserves dtype."""
        other = float32_raster.model_copy()
        result = float32_raster + other
        assert result.arr.dtype == np.float32

    def test_multiplication_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that multiplication preserves float32."""
        result = float32_raster * 2.0
        assert result.arr.dtype == np.float32

    def test_multiplication_preserves_dtype_float64(self, float64_raster: Raster):
        """Test that multiplication preserves float64."""
        result = float64_raster * 2.0
        assert result.arr.dtype == np.float64

    def test_subtraction_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that subtraction preserves float32."""
        result = float32_raster - 1.0
        assert result.arr.dtype == np.float32

    def test_division_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that division preserves float32."""
        result = float32_raster / 2.0
        assert result.arr.dtype == np.float32

    def test_negation_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that negation preserves float32."""
        result = -float32_raster
        assert result.arr.dtype == np.float32

    def test_apply_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that apply() preserves dtype."""
        result = float32_raster.apply(lambda x: x * 2)
        assert result.arr.dtype == np.float32

    def test_apply_raw_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that apply() with raw=True preserves dtype."""
        result = float32_raster.apply(lambda arr: arr * 2, raw=True)
        assert result.arr.dtype == np.float32

    def test_fillna_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that fillna() preserves dtype."""
        # Add a NaN value
        raster_with_nan = float32_raster.model_copy()
        raster_with_nan.arr[0, 0] = np.nan
        result = raster_with_nan.fillna(0.0)
        assert result.arr.dtype == np.float32

    def test_normalize_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that normalize() preserves dtype."""
        result = float32_raster.normalize()
        assert result.arr.dtype == np.float32

    def test_blur_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that blur() preserves dtype."""
        result = float32_raster.blur(sigma=0.5)
        assert result.arr.dtype == np.float32

    def test_extrapolate_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that extrapolate() preserves dtype."""
        # Add NaN values to test extrapolation
        raster_with_nan = float32_raster.model_copy()
        raster_with_nan.arr[0, 0] = np.nan
        result = raster_with_nan.extrapolate()
        assert result.arr.dtype == np.float32

    def test_pad_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that pad() preserves dtype."""
        result = float32_raster.pad(width=1.0)
        assert result.arr.dtype == np.float32

    def test_crop_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that crop() preserves dtype."""
        bounds = (0.0, 0.0, 1.5, 1.5)
        result = float32_raster.crop(bounds)
        assert result.arr.dtype == np.float32

    def test_taper_border_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that taper_border() preserves dtype."""
        result = float32_raster.taper_border(width=0.5)
        assert result.arr.dtype == np.float32

    def test_trim_nan_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that trim_nan() preserves dtype."""
        # Add NaN values around edges
        raster_with_nan = float32_raster.pad(width=1.0, value=np.nan)
        result = raster_with_nan.trim_nan()
        assert result.arr.dtype == np.float32

    def test_resample_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that resample() preserves dtype."""
        result = float32_raster.resample(new_cell_size=0.5)
        assert result.arr.dtype == np.float32

    def test_model_copy_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that model_copy() preserves dtype."""
        result = float32_raster.model_copy()
        assert result.arr.dtype == np.float32

    def test_statistical_methods_return_float(self, float32_raster: Raster):
        """Test that statistical methods return Python floats."""
        # These should all return float, not numpy types
        assert isinstance(float32_raster.max(), float)
        assert isinstance(float32_raster.min(), float)
        assert isinstance(float32_raster.mean(), float)
        assert isinstance(float32_raster.std(), float)
        assert isinstance(float32_raster.median(), float)
        assert isinstance(float32_raster.quantile(0.5), float)
