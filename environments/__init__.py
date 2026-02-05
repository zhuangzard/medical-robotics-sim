"""
Environments package for medical-robotics-sim

Available environments:
- PushBoxEnv: Simple rigid body manipulation task

Author: Physics-Informed Robotics Team
Date: 2026-02-05
"""

from environments.push_box import PushBoxEnv, make_push_box_env

__all__ = [
    'PushBoxEnv',
    'make_push_box_env'
]

__version__ = '0.1.0'
