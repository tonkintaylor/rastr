import pytest
from shapely.geometry import LineString, MultiLineString, Point

from rastr.gis.cast import cast_multilinestring


class TestCastMultiLineString:
    def test_cast_multilinestring(self):
        # Arrange
        multiline = MultiLineString(
            [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]
        )

        # Act
        result = multiline

        # Assert
        assert isinstance(result, MultiLineString)
        assert result.equals(multiline)

    def test_cast_linestring(self):
        # Arrange
        line = LineString([(0, 0), (1, 1)])

        # Act
        result = cast_multilinestring(line)

        # Assert
        assert isinstance(result, MultiLineString)
        expected = MultiLineString([line])
        assert result.equals(expected)

    def test_cast_invalid_geometry_raises(self):
        # Arrange
        point = Point(0, 0)

        # Act / Assert
        with pytest.raises(
            TypeError, match=r"Cannot cast geometry of type Point to MultiLineString."
        ):
            _ = cast_multilinestring(point)
