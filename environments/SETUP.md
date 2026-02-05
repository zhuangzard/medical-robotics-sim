# PushBox Environment Setup Guide

## Prerequisites

The PushBox environment requires:
- Python 3.10+
- MuJoCo 2.3.7+
- Gymnasium (OpenAI Gym successor)
- NumPy

## Installation Options

### Option 1: Using Conda (Recommended)

```bash
# Create environment from yaml
cd ~/.openclaw/workspace/medical-robotics-sim
conda env create -f environment.yml
conda activate physics-robot

# Verify installation
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
```

### Option 2: Using pip + venv

```bash
cd ~/.openclaw/workspace/medical-robotics-sim

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install numpy>=1.23.0
pip install mujoco>=2.3.7
pip install gymnasium>=0.28.0
pip install torch torchvision torchaudio

# Verify
python -c "import mujoco; print('MuJoCo OK')"
python -c "import gymnasium; print('Gymnasium OK')"
```

## Quick Test

After installation, verify the environment works:

```bash
cd ~/.openclaw/workspace/medical-robotics-sim

# Run test suite
python environments/test_push_box.py

# Test baseline controllers
python baselines/simple_controller.py

# Test data schema
python data/data_schema.py
```

## Expected Output

All tests should pass:
```
✓ PASS: Initialization
✓ PASS: Random Policy
✓ PASS: Mass Variation
✓ PASS: Rendering
✓ PASS: Episode Data
✓ PASS: Success Condition

Total: 6/6 tests passed
🎉 All tests passed!
```

## Troubleshooting

### MuJoCo not found

```bash
pip install mujoco>=2.3.7
```

### Gymnasium not found

```bash
pip install gymnasium>=0.28.0
```

### Import errors

Make sure you're in the project root directory:
```bash
cd ~/.openclaw/workspace/medical-robotics-sim
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Rendering issues (macOS)

If you get OpenGL errors, try:
```bash
pip install pyopengl
```

For headless rendering:
```python
env = PushBoxEnv(render_mode='rgb_array')  # Use rgb_array instead of human
```

## File Structure

```
environments/
├── __init__.py           # Package exports
├── push_box.py          # Main environment (~400 lines)
├── test_push_box.py     # Test suite (~200 lines)
├── assets/
│   └── push_box.xml     # MuJoCo scene definition
└── SETUP.md             # This file

baselines/
├── __init__.py          # Controller exports
└── simple_controller.py # P, PD, Greedy, Random controllers

data/
├── data_schema.py       # Episode and experiment data structures
└── (experiment results will be saved here)
```

## Next Steps

1. Run test suite to verify installation
2. Benchmark baseline controllers
3. Integrate with Dynami-CAL physics learning
4. Collect data for paper experiments

## Paper Experiments

For Section 4.1 validation:

```bash
# Baseline (MuJoCo only)
python experiments/run_baseline.py --episodes 5000

# Dynami-CAL (physics-informed)
python experiments/run_dynamical.py --episodes 400

# OOD testing
python experiments/run_ood.py --masses 0.5 1.0 1.5 2.0
```

Expected results (paper claims):
- Sample efficiency: 12.5x (400 vs 5000 episodes)
- OOD generalization: < 20% performance drop at 2x mass

## Contact

Project: Medical Robotics Simulation
Team: Physics-Informed Robotics
Date: 2026-02-05
