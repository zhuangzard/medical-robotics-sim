"""
Baseline controllers for medical-robotics-sim

Available controllers:
- ProportionalController: Simple P-controller
- PDController: PD-controller with damping
- GreedyController: Always move toward goal
- RandomController: Random actions

Author: Physics-Informed Robotics Team
Date: 2026-02-05
"""

from baselines.simple_controller import (
    ProportionalController,
    PDController,
    GreedyController,
    RandomController,
    evaluate_controller
)

__all__ = [
    'ProportionalController',
    'PDController',
    'GreedyController',
    'RandomController',
    'evaluate_controller'
]

__version__ = '0.1.0'
