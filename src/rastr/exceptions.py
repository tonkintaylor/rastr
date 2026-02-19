"""Custom exceptions for the rastr package."""


class NonSquareCellsError(ValueError):
    """Raised when square cells are required but the raster has non-square cells."""
