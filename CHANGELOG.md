# CHANGELOG - PhysRobot Project

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### To Do
- Fix training timestep counter (P0 blocker)
- Restore physics core in Colab notebook v4
- Run 3-seed pilot experiments
- Generate real training curves

---

## [0.2.0] - 2026-02-09

### 🎨 Refactored - Complete Project Reorganization

**Major restructuring** to prepare for paper submission and team collaboration.

#### Added
- **PROJECT_BOOK.md** (21KB) - Comprehensive project documentation
  - 6 major sections: Vision, Innovations, Learning, Deliverables, Issues, Roadmap
  - Complete technical descriptions of all 4 innovations
  - Reference to 8 key papers with context
  - Detailed roadmap through ICRA 2027 submission
  
- **HANDOFF.md** (9KB) - Agent collaboration protocol
  - Current task tracking
  - Blocker documentation
  - Context for next agent
  - Session metadata
  
- **CHANGELOG.md** (this file) - Progress tracking
  
- **archive/** directory - Moved all process/temp files here (14 documents)

#### Changed
- **README.md** - Complete rewrite
  - Professional format suitable for paper repo
  - Added abstract, innovations table, comparison matrix
  - Added current status dashboard
  - Added roadmap with 4 phases
  - Added planned experiments section
  
- **.gitignore** - Updated patterns
  - Added `research/dynamical-gnn/source/.venv/` (prevent huge venv)
  - Added `**/__pycache__/` (recursive pattern)
  - Added `*.pyc` and `firebase-debug.log`

#### Moved
- **src/** - All production code organized here
  - `physics_core/` ← physics_core/
  - `environments/` ← environments/
  - `baselines/` ← baselines/
  - `training/` ← training/
  
- **notebooks/** - All Colab notebooks
  - Moved from colab/* (18 files)
  - Includes v1, v2, v3 training notebooks + ablation
  
- **archive/** - Process/planning documents
  - COMPLETION_REPORT.md
  - DAY1_2_COMPLETION_REPORT.md
  - PUSHBOX_COMPLETION_REPORT.md
  - PHYSROBOT_DIAGNOSIS.md
  - FIXES_VERIFIED.md
  - VERIFICATION_GUIDE.md
  - WEEK1_DAY5_HANDOFF.md
  - WEEK1_PROGRESS.md
  - TECHNICAL_SURVEY.md
  - MEDICAL_ROBOTICS_REQUIREMENTS.md
  - PROJECT_ROADMAP.md
  - PROTOTYPE_PLAN.md
  - PROJECT_STRUCTURE.txt
  - EXPERIMENT_LOG.md

#### Git Stats
- **64 files renamed/moved** (using `git mv` for history preservation)
- **3 new files created** (PROJECT_BOOK.md, HANDOFF.md, CHANGELOG.md)
- **1 file updated** (.gitignore)
- **Total changes**: 68 file operations

---

## [0.1.0] - 2026-02-06

### 🎯 Week 1 Completion - Core Implementation

#### Added
- **physics_core/** modules (800+ lines total)
  - `edge_frame.py` (203 lines) - EdgeFrame with antisymmetry
  - `sv_message_passing.py` (187 lines) - Scalar-Vector message passing
  - `dynamical_gnn.py` (312 lines) - Full Dynami-CAL GNN architecture
  - `integrators.py` (156 lines) - Symplectic & RK4 integrators
  
- **tests/** - Unit test suite
  - `test_edge_frame.py` - Antisymmetry and equivariance tests
  - `test_conservation.py` - Momentum and energy conservation
  - `test_sv_message_passing.py` - Scalar-vector decomposition
  - **Results**: 21 passing, 12 skipped (need GPU)
  
- **environments/push_box.py** (428 lines)
  - MuJoCo-based 2D pushing environment
  - Gym API compliant
  - Configurable mass, friction, goal positions
  
- **baselines/** - 4 comparison methods
  - `ppo_baseline.py` - Pure RL baseline
  - `gns_baseline.py` - Graph Network Simulator
  - `hnn_baseline.py` - Hamiltonian Neural Network
  - `physics_informed.py` - Our physics-informed approach
  
- **training/** - Training infrastructure
  - `config.yaml` - Hyperparameter configuration
  - `train.py` - Main training loop (⚠️ has bugs)
  - `eval.py` - Evaluation script
  
- **colab/** - Google Colab notebooks
  - `week1_full_training_v1.ipynb` - Initial version (working)
  - `week1_full_training_v2.ipynb` - Improved logging
  - `week1_full_training_v3.ipynb` - Simplified (⚠️ removed physics core by mistake)
  - `phase1_ablation.ipynb` - Ablation study template

#### Fixed
- EdgeFrame antisymmetry error: **< 1e-5** (verified)
- Conservation error: **< 0.1%** for momentum, **< 1%** for angular momentum
- Symplectic energy drift: **< 0.01%** per 1000 steps

#### Known Issues (from Week 1)
- ⚠️ Training timesteps stuck at 16K / 200K (not incrementing)
- ⚠️ Colab notebook v3 simplified away physics core
- ⚠️ Missing torch_geometric and mujoco on local machine

---

## [0.0.2] - 2026-02-05

### 📚 Research & Planning Phase

#### Added
- **research/** directory structure
  - `dynamical-gnn/` - 14-chapter tutorial on physics-informed GNN
  - `paper_drafts/` - Paper draft V1 (IEEE format, 8 pages)
  - Literature review documents
  
- **docs/单项学习/** - Learning materials
  - GeometricDL/ - 7-chapter geometric deep learning guide
  - MuJoCo/ - MuJoCo tutorial notes
  - PPO/ - PPO algorithm study
  - GNS/ - Graph Network Simulator notes
  
- Planning documents (now in archive/)
  - TECHNICAL_SURVEY.md - Physics engine comparison
  - MEDICAL_ROBOTICS_REQUIREMENTS.md - Application requirements
  - PROJECT_ROADMAP.md - Initial roadmap
  - PROTOTYPE_PLAN.md - Implementation plan
  - PROOF_OF_CONCEPT_PLAN.md - PoC strategy

#### Research Completed
- Literature review: 8 key papers (EGNN, GNS, HNN, DimeNet, etc.)
- Physics engine survey: 7 engines compared
- Medical robotics requirements analysis
- Innovation design: 4 core innovations identified

---

## [0.0.1] - 2026-02-04

### 🚀 Project Initialization

#### Added
- Initial repository structure
- `environment.yml` - Conda environment specification
- `requirements.txt` - Python dependencies
- `.gitignore` - Basic ignore patterns
- `README.md` - Initial project description
- `GIT_WORKFLOW.md` - Git conventions
- `MILESTONES.md` - High-level milestones

#### Set Up
- Git repository initialized
- Conda environment created (`physics-robot`)
- Basic directory structure: data/, models/, results/, experiments/

---

## Version History Summary

| Version | Date | Focus | Status |
|---------|------|-------|--------|
| **0.2.0** | 2026-02-09 | Project reorganization | ✅ Complete |
| **0.1.0** | 2026-02-06 | Core implementation | ✅ Code done, training broken |
| **0.0.2** | 2026-02-05 | Research & planning | ✅ Complete |
| **0.0.1** | 2026-02-04 | Project initialization | ✅ Complete |

---

## Upcoming Milestones

### [0.3.0] - Training Pipeline Fix (Target: Week 3, Feb 2026)

**Goal**: Get training working end-to-end (16K → 200K timesteps)

#### Planned
- [ ] Debug timestep counter in train.py
- [ ] Fix Colab notebook v4 (restore physics core)
- [ ] Add debug logging to training loop
- [ ] Verify conservation laws during training
- [ ] Run 3-seed pilot experiments (10K timesteps each)

**Acceptance Criteria**:
- Training completes 200K timesteps successfully
- Conservation error < 0.5% throughout training
- Checkpoints saved every 10K steps
- W&B logging shows smooth loss curves

---

### [0.4.0] - Baseline Experiments (Target: Week 5-6, Mar 2026)

**Goal**: Run all 4 baselines × 5 seeds, generate comparison data

#### Planned
- [ ] PPO baseline (5 seeds × 200K steps)
- [ ] GNS baseline (5 seeds × 200K steps)
- [ ] HNN baseline (5 seeds × 200K steps)
- [ ] PhysRobot (5 seeds × 200K steps)
- [ ] Generate training curves (with error bars)
- [ ] Compute sample efficiency metrics

**Acceptance Criteria**:
- All 20 runs (4 methods × 5 seeds) complete
- Mean ± std computed for all metrics
- PhysRobot shows ≥10× sample efficiency (conservative)

---

### [0.5.0] - OOD Generalization & Ablations (Target: Week 7-8, Mar 2026)

**Goal**: Test generalization and ablate innovations

#### Planned
- [ ] OOD experiments: test on 5 different masses
- [ ] Ablation 1: Remove EdgeFrame antisymmetry
- [ ] Ablation 2: Remove SV-message passing
- [ ] Ablation 3: Remove symplectic integrator
- [ ] Ablation 4: Full model (baseline)
- [ ] Generate ablation bar chart

**Acceptance Criteria**:
- OOD performance ≥90% (PhysRobot vs baselines)
- Each ablation shows measurable performance drop
- Conservation error increases with each removed component

---

### [1.0.0] - Paper Submission Ready (Target: Week 11-12, May 2026)

**Goal**: Complete paper with real data, ready for CoRL 2026

#### Planned
- [ ] Update all figures with real data
- [ ] Rewrite experiment section
- [ ] Add statistical significance tests
- [ ] Internal review (3 readers)
- [ ] Proofread and polish
- [ ] Prepare supplementary material (code release)
- [ ] Create demo video (1 minute)

**Acceptance Criteria**:
- All claims backed by experimental data
- Error bars on all plots (std over 5 seeds)
- Statistical tests (p < 0.05) for all comparisons
- Code repository public-ready (clean, documented)
- Submission 1 week before deadline (buffer)

---

## Contributing

When making changes:

1. Update this CHANGELOG under `[Unreleased]`
2. Move changes to a new version section on release
3. Update HANDOFF.md with current status
4. Link to issues/PRs where applicable

**Version numbering**:
- **Major (X.0.0)**: Paper submission, major milestones
- **Minor (0.X.0)**: Feature additions, significant experiments
- **Patch (0.0.X)**: Bug fixes, small improvements

---

**Last Updated**: 2026-02-09  
**Maintained By**: PhysRobot Team  
**Next Review**: After training pipeline fix (Week 3)
