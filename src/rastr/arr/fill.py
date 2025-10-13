from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def fillna_nearest_neighbours(arr: NDArray) -> NDArray:
    """Fill NaN values in an N-dimensional array with their nearest neighbours' values.

    The nearest neighbour is determined using the Euclidean distance between array
    indices, so there is equal weighting given in all directions (i.e. across all axes).
    In the case of tiebreaks, the value from the neighbour with the lowest index is
    imputed.
    """
    from scipy.interpolate import NearestNDInterpolator

    nonnan_mask = np.nonzero(~np.isnan(arr))
    nonnan_idxs = np.array(nonnan_mask).transpose()

    if nonnan_idxs.size == 0:
        # Everything is NaN
        return arr

    # Interpolate at the array indices
    interp = NearestNDInterpolator(nonnan_idxs, arr[nonnan_mask])
    filled_arr = interp(*np.indices(arr.shape))
    # Preserve the original dtype
    return filled_arr.astype(arr.dtype)
