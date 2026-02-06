"""
Environments package for medical-robotics-sim

Canonical environments:
- PushBoxEnv:          2-DOF single-box push (16-dim obs, 2-dim action)
- MultiObjectPushEnv:  Point-mass multi-object push (variable obs, 2-dim action)

See ENV_SPECIFICATION.md for full parameter documentation.
"""

from environments.push_box import PushBoxEnv, make_push_box_env
from environments.multi_object_push import MultiObjectPushEnv, make_multi_push_env

__all__ = [
    'PushBoxEnv',
    'make_push_box_env',
    'MultiObjectPushEnv',
    'make_multi_push_env',
]

__version__ = '0.2.0'
