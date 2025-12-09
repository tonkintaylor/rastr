"""Deprecated: Use rastr.io_ instead.

This module provides backwards compatibility by re-exporting all functions from io_.
"""  # noqa: A005

import warnings

warnings.warn(
    "rastr.io is deprecated. Import from rastr.io_ instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

from rastr.io_ import *  # noqa: F403, E402
