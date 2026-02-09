# Experiment Log

**Project**: Physics-Informed Medical Robotics  
**Goal**: ICRA 2027 / CoRL 2026 Paper  
**Key Metrics**: 12.5× sample efficiency, 95% OOD generalization

---

## Week 1: PushBox Environment

### Day 1-2: Environment Setup ✅

**Date**: 2026-02-05  
**Duration**: 4-5 hours  
**Status**: ✅ Complete

#### Tasks Completed

- [x] Project structure created
- [x] Conda environment configured (`physics-robot`)
- [x] Core modules implemented:
  - [x] `edge_frame.py` (200 lines)
  - [x] `dynamical_gnn.py` (300 lines)
  - [x] `integrators.py` (100 lines)
- [x] Unit tests written (200 lines)
- [x] Configuration files (`config.yaml`)
- [x] Documentation (`README.md`, `EXPERIMENT_LOG.md`)
- [x] Git initialization

#### Verification Results

| Test | Target | Actual | Status |
|------|--------|--------|--------|
| EdgeFrame antisymmetry | < 1e-5 | TBD | ⏳ Pending |
| Conservation error | < 0.1% | TBD | ⏳ Pending |
| Dependencies installed | All | TBD | ⏳ Pending |

**Next Steps**: Run unit tests to verify all modules

---

### Day 3-4: PushBox Environment (Planned)

**Target Date**: 2026-02-06 - 2026-02-07  
**Estimated Duration**: 6-8 hours

#### Tasks

- [ ] Implement PushBox Gym environment
- [ ] Create baseline PPO agent
- [ ] Implement physics-informed training loop
- [ ] Data collection pipeline
- [ ] Logging and visualization

#### Success Criteria

- [ ] Environment renders correctly
- [ ] Baseline PPO trains successfully (10K timesteps)
- [ ] Physics-informed agent trains (800 timesteps)
- [ ] Conservation laws maintained during rollout

---

### Day 5-7: Experiments & Data Collection (Planned)

**Target Date**: 2026-02-08 - 2026-02-10

#### Experiments

1. **Sample Efficiency Test**
   - Baseline: Pure PPO (10,000 timesteps)
   - Physics: Dynami-CAL GNN (800 timesteps)
   - Compare success rates

2. **Conservation Law Analysis**
   - Track energy drift over 100 steps
   - Compare Symplectic vs RK4 integrators

3. **Ablation Study**
   - w/ EdgeFrame vs w/o EdgeFrame
   - w/ Conservation loss vs w/o
   - Different message passing layers (1, 2, 3, 5)

#### Data Format

```yaml
experiment_id: exp_001
timestamp: 2026-02-08T10:30:00
config:
  algorithm: PPO
  timesteps: 10000
  hidden_dim: 128
  
results:
  success_rate: 0.85
  final_reward: 120.5
  energy_drift: 0.0023
  training_time: 3600  # seconds
```

---

## Week 2: Tissue Grasping (Planned)

### Day 8-10: Environment Development

- [ ] Implement soft tissue simulation
- [ ] Force control interface
- [ ] Multi-object grasping scenarios

### Day 11-14: OOD Generalization Test

**Key Test**: Train on Object A, test on Object B

- [ ] Train on rubber ball (stiffness = 1.0)
- [ ] Test on foam cube (stiffness = 0.5)
- [ ] Target: 95% performance retention

---

## Week 3-4: Paper Writing (Planned)

### Figures to Generate

1. Sample efficiency curve (PPO vs Physics)
2. Conservation law plot (energy over time)
3. OOD generalization bar chart
4. Ablation study table

### Tables to Generate

1. Hyperparameter summary
2. Quantitative results comparison
3. Conservation error statistics

---

## Debugging Notes

### Common Issues

1. **PyTorch Geometric Installation**
   - Issue: torch-scatter compilation errors
   - Solution: Use `cpuonly` on Mac M1/M2

2. **MuJoCo License**
   - Issue: License not found
   - Solution: MuJoCo 2.x is free, no license needed

3. **Memory Issues**
   - Issue: OOM during training
   - Solution: Reduce batch size or hidden_dim

---

## Data Collection Format

### Training Metrics

Record every 100 steps:

```python
{
    "step": 100,
    "episode_reward": 85.3,
    "success_rate": 0.75,
    "energy_error": 0.0012,
    "momentum_error": 0.0008,
    "loss": 0.023,
    "learning_rate": 0.0003,
}
```

### Evaluation Metrics

Record after each evaluation (every 500 steps):

```python
{
    "eval_step": 500,
    "eval_episodes": 10,
    "mean_reward": 92.1,
    "std_reward": 8.5,
    "success_rate": 0.80,
    "mean_episode_length": 45,
}
```

---

## Conservation Analysis Template

### Energy Conservation

```
Initial Energy: E₀ = 10.5 J
Final Energy: E₁ = 10.52 J
Absolute Error: ΔE = 0.02 J
Relative Error: ΔE/E₀ = 0.19%
Status: ✅ PASS (< 0.1% target)
```

### Momentum Conservation

```
Initial Momentum: p₀ = [1.2, 0.5, 0.0] kg⋅m/s
Final Momentum: p₁ = [1.21, 0.51, 0.01] kg⋅m/s
Error: ||Δp|| = 0.014 kg⋅m/s
Relative Error: 1.1%
Status: ⚠️ WARNING (> 0.1% target)
```

---

## Results Summary Template

### Week 1 Results

| Metric | Baseline PPO | Physics-Informed | Improvement |
|--------|--------------|------------------|-------------|
| Success Rate | 85% | 90% | +5% |
| Sample Efficiency | 10,000 steps | 800 steps | **12.5×** |
| Training Time | 2 hours | 15 min | 8× faster |
| Energy Drift | 5.2% | 0.08% | **65× better** |

---

## Notes

- Use W&B for experiment tracking: `wandb.init(project="physics-robot-week1")`
- Save checkpoints every 1000 steps
- Log trajectories for visualization
- Record hyperparameters in `config.yaml`

---

## Citations

Add relevant papers as experiments progress.

**Last Updated**: 2026-02-05
