"""
DEPRECATED — this module is a shim that re-exports from push_box.py.
All code should import directly from environments.push_box.

This file exists only to avoid breaking older imports during the transition.
It will be removed in a future release.
"""

import warnings as _warnings
_warnings.warn(
    "environments.push_box_env is deprecated. "
    "Import from environments.push_box instead.",
    DeprecationWarning,
    stacklevel=2,
)

from environments.push_box import PushBoxEnv, make_push_box_env  # noqa: F401
