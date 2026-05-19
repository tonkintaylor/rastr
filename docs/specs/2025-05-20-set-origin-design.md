# Spec: `set_origin` — Relocate Raster Grid Origin

**Date:** 2025-05-20
**Issue:** #442
**Branch:** `442-feature-request-shift_bounds-translate_bounds-method-for-longitude-wrapping`

## Motivation

Some upstream data sources produce raster files with non-standard longitude
conventions. For example, certain ShakeMap TIFs for events near the
antimeridian (e.g. New Zealand) store longitudes as negative values that
exceed -180° (e.g. bounds from -200° to -170° instead of 160° to 190°).

This is a data-quality issue in the source files, not a bug in rastr. However,
when such a raster is loaded, spatial operations like `to_bounds()` produce
all-NaN output because the stored coordinates have no overlap with target
bounds expressed in standard EPSG:4326.

The existing workaround — manually reconstructing the affine transform or
rebuilding the Raster with adjusted metadata — works but is verbose and
non-obvious. The purpose of this feature is not to fix the upstream data
problem, but to provide a clean, readable API for callers who need to
normalize such rasters before performing spatial operations.

## Design

### 1. Read-only property: `Raster.origin`

```python
@property
def origin(self) -> tuple[float, float]:
    """The grid origin (x, y) — the corner of the first pixel.

    This is the translation component (c, f) of the affine transform.
    For north-up rasters this corresponds to (xmin, ymax); for south-up
    rasters it corresponds to (xmin, ymin).
    """
    return (self.transform.c, self.transform.f)
```

### 2. Property setter (mutable, for notebooks/interactive use)

```python
@origin.setter
def origin(self, value: tuple[float, float]) -> None:
    """Set the grid origin via the transform."""
    x, y = value
    t = self.transform
    self.transform = Affine(t.a, t.b, x, t.d, t.e, y)
```

### 3. Immutable method: `Raster.set_origin()`

```python
def set_origin(self, *, x: float | None = None, y: float | None = None) -> Self:
    """Set the transform origin without modifying pixel data.

    The origin is the corner of the first pixel in the raster grid —
    the translation component (c, f) of the affine transform. For
    north-up rasters this corresponds to (xmin, ymax); for south-up
    rasters it corresponds to (xmin, ymin).

    Unspecified axes retain their current value. If neither axis is
    provided, returns an unchanged copy.

    Args:
        x: New x-coordinate of the grid origin. If None, keeps current.
        y: New y-coordinate of the grid origin. If None, keeps current.

    Returns:
        A new Raster with the updated transform origin and unchanged array data.

    Example:
        Normalize a raster with non-standard longitude (e.g., -200° to -170°)
        to standard EPSG:4326 range::

            raster = raster.set_origin(x=raster.origin[0] + 360)
    """
```

## Interface Decisions

| Decision | Rationale |
|----------|-----------|
| Keyword-only args on `set_origin()` | Passing one axis alone is common; `set_origin(160)` would be ambiguous about which axis |
| `None` = keep current | Allows shifting just x or just y without computing the other |
| No-op when both are None | Not an error — returns an unchanged copy |
| Both property setter and method | Setter for interactive/notebook mutation; method for immutable pipeline use. Mirrors the existing `crs`/`set_crs()` pattern |
| No `set_transform()` method | The `transform` property setter already covers arbitrary replacement. A method would just add `allow_override` ceremony with no value since a transform is always present |
| No delta/shift method | `raster.origin[0] + 360` is simple enough. Can revisit if arithmetic at call sites becomes tedious |
| Docstring acknowledges both axis conventions | "Corner of the first pixel" is unambiguous regardless of north-up vs south-up |

## Placement

- `origin` property + setter: next to the existing `transform` property/setter block (~line 136 in `raster.py`)
- `set_origin()` method: next to `set_crs()` (~line 427 in `raster.py`)

## Usage Example

```python
from rastr import Raster

raster = Raster.read_file("shakemap.tif")

# Raster has non-standard longitude: bounds are (-200, -50, -170, -30)
if raster.bounds.xmin < -180:
    raster = raster.set_origin(x=raster.origin[0] + 360)

# Now bounds are (160, -50, 190, -30) — standard range
result = raster.to_bounds(target_bounds)
```

## Deliberately Excluded

The following were considered and rejected as contrary to the design intent:

- **Automatic longitude detection/normalization** — Baking a detection heuristic into the library is premature. Different datasets use different conventions (0–360, -180–180, etc.) and the "correct" normalization depends on the caller's target coordinate space. The caller decides when and how much to shift.
- **Reprojection or pixel resampling** — This feature is metadata-only by design. It adjusts where the grid sits in coordinate space without touching pixel values. Reprojection is a fundamentally different operation.
- **Validation that the new origin is "sensible"** — The user knows their data. Adding bounds-checking would prevent legitimate use cases (e.g. rasters in projected CRS with large coordinate values) without catching the errors it aims to prevent.
