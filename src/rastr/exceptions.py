"""Custom exceptions for the rastr package."""


class RastrError(Exception):
    """Base exception for the rastr package."""


class NonSquareCellsError(ValueError, RastrError):
    """Raised when square cells are required but the raster has non-square cells."""
