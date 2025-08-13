import pytest
from shapely.geometry import Polygon

from rastr.gis.fishnet import (
    create_fishnet,
    create_point_grid,
    get_point_grid_shape,
)


def _normalize_polygon(polygon):
    """Normalize a polygon's coordinates by sorting them."""
    coords = sorted(
        polygon.exterior.coords[:-1]
    )  # Sort the coordinates, exclude the closing point
    return Polygon(coords)


def assert_polygons_equal(
    result: list[Polygon], expected: list[Polygon], tol: float = 1e-9
) -> None:
    """Assert that two lists of polygons are equal."""
    if len(result) != len(expected):
        pytest.fail(f"Expected {len(expected)} geometries, got {len(result)}")

    normalized_result = [_normalize_polygon(p) for p in result]
    normalized_expected = [_normalize_polygon(p) for p in expected]

    for res_poly in normalized_result:
        if not any(
            res_poly.equals_exact(exp_poly, tol) for exp_poly in normalized_expected
        ):
            pytest.fail(f"Geometry {res_poly} not found in expected geometries")


class TestFishnet:
    def test_regular_example(self):
        bounds = (0.0, 0.0, 2.0, 2.0)
        expected_polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
        ]

        result = create_fishnet(bounds=bounds, res=1.0)

        assert_polygons_equal(result, expected_polygons)

    def test_res_not_factor_of_bounds_dims(self):
        bounds = (0.0, 0.0, 1.0, 2.0)
        expected_polygons = [
            Polygon([(0, 2), (0.6, 2), (0.6, 1.4), (0, 1.4)]),
            Polygon([(0.6, 2), (1.2, 2), (1.2, 1.4), (0.6, 1.4)]),
            Polygon([(0, 1.4), (0.6, 1.4), (0.6, 0.8), (0, 0.8)]),
            Polygon([(0.6, 1.4), (1.2, 1.4), (1.2, 0.8), (0.6, 0.8)]),
            Polygon([(0, 0.8), (0.6, 0.8), (0.6, 0.2), (0, 0.2)]),
            Polygon([(0.6, 0.8), (1.2, 0.8), (1.2, 0.2), (0.6, 0.2)]),
            Polygon([(0, 0.2), (0.6, 0.2), (0.6, -0.4), (0, -0.4)]),
            Polygon([(0.6, 0.2), (1.2, 0.2), (1.2, -0.4), (0.6, -0.4)]),
        ]
        result = create_fishnet(bounds=bounds, res=0.6)

        assert_polygons_equal(result, expected_polygons)


class TestGetPointGridShape:
    def test_matches_create_point_grid_simple(self):
        bounds = (0.0, 0.0, 2.0, 2.0)
        cell_size = 1.0
        expected_shape = create_point_grid(bounds=bounds, cell_size=cell_size)[0].shape

        shape = get_point_grid_shape(bounds=bounds, cell_size=cell_size)

        assert shape == expected_shape

    def test_matches_create_point_grid_borderline(self):
        bounds = (0.0, 0.0, 2.0, 2.0)
        cell_size = 1.5
        expected_shape = create_point_grid(bounds=bounds, cell_size=cell_size)[0].shape

        shape = get_point_grid_shape(bounds=bounds, cell_size=cell_size)

        assert shape == expected_shape
