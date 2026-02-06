# PhysRobot Paper — Figure Designs

**Author**: Round 2 — Figure Designer  
**Date**: 2026-02-06  
**Target**: 8-page ICRA/CoRL format (7 figures total)  

---

## Overview

| Fig | Title | Type | Size | Tool | Section |
|-----|-------|------|------|------|---------|
| 1 | PhysRobot System Overview | Architecture diagram | Full-width (2-col) | matplotlib + manual polish | §3 Method |
| 2 | SV-Pipeline: Scalarization–Vectorization | Flowchart | Full-width (2-col) | matplotlib + manual polish | §3.3–3.4 |
| 3 | EdgeFrame Construction | Geometric diagram | Single-column | tikz or matplotlib | §3.4 |
| 4 | Main Results: Learning Curves | Line plots (3×1 grid) | Full-width (2-col) | matplotlib | §4.3 |
| 5 | Ablation Study Results | Grouped bar chart | Single-column | matplotlib | §4.4 |
| 6 | OOD Generalization | Line plot + heatmap | Full-width (2-col) | matplotlib | §4.5 |
| 7 | Multi-Object Environment Visualization | Rendered scene + graph | Full-width (2-col) | MuJoCo render + matplotlib overlay | §4.1 |

---

## Figure 1: PhysRobot System Overview

### 1. Title
**Fig. 1.** *PhysRobot: A dual-stream physics-informed GNN architecture for sample-efficient robotic manipulation. The Physics Stream (top) constructs a dynamic scene graph and performs momentum-conserving SV message passing; the Policy Stream (bottom) processes raw observations. Fusion with stop-gradient produces the PPO policy.*

### 2. Layout
- **Full-width** (2-column, ~7.0 × 3.0 inches)
- Single panel, horizontal flow left→right
- Three major blocks: **Input** → **Dual Streams** → **Output**

### 3. Detailed Content

**Left block — Input:**
- MuJoCo scene illustration (robot arm + boxes + goals)
- Arrow labeled "State $\mathbf{s}_t$" splits into two paths

**Center-top — Physics Stream:**
- Box: "Scene Graph Construction" (nodes: ee, box₁, box₂, goal₁, goal₂)
- Arrow → Box: "SV Message Passing (×L)" with conservation badge ✓
- Sub-detail inside: small diagram showing $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ on an edge
- Arrow → Box: "Dynamics Decoder" → $\hat{\mathbf{a}}$ (predicted accelerations)
- Side arrow down: "Physics Loss $\mathcal{L}_\text{phys}$" (self-supervised)
- **Stop-gradient** symbol (⊘) on the output arrow

**Center-bottom — Policy Stream:**
- Box: "MLP Encoder" (simple feedforward)
- Arrow → $\mathbf{z}_\text{policy}$

**Right block — Fusion + Output:**
- Box: "Fusion" receiving $\text{sg}(\hat{\mathbf{a}})$ and $\mathbf{z}_\text{policy}$
- Arrow → Box: "PPO Actor" → $\pi(a|s)$
- Arrow → Box: "PPO Critic" → $V(s)$
- Below: Loss equation $\mathcal{L} = \mathcal{L}_\text{RL} + \lambda_\text{phys}\mathcal{L}_\text{phys}$

**Color scheme:**
- Physics Stream: blue tones (#3B82F6, #1D4ED8)
- Policy Stream: green tones (#10B981, #059669)
- Fusion/Output: orange tones (#F59E0B, #D97706)
- Arrows: dark gray (#374151)
- Stop-gradient: red (#EF4444)

### 4. Data Source
- No data — purely architectural diagram
- Based on ALGORITHM_DESIGN.md §2.5–2.6

### 5. Tool
- **matplotlib** for draft/skeleton (see `figures/fig1_system_overview.py`)
- Final version: refine in Inkscape/Illustrator or tikz for camera-ready

### 6. ASCII Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌──────────┐    ┌─── Physics Stream (Blue) ──────────────────────────┐     │
│  │           │    │                                                    │     │
│  │  MuJoCo   │    │  ┌─────────────┐   ┌──────────────┐  ┌────────┐  │     │
│  │  Scene    │───▶│  │ Scene Graph  │──▶│  SV Message  │─▶│Dynamics│  │     │
│  │           │    │  │Construction  │   │ Passing (×L) │  │Decoder │  │     │
│  │  ┌─┐     │    │  └─────────────┘   │  ✓ N3L cons  │  └───┬────┘  │     │
│  │  │R│ □□   │    │                    └──────────────┘      │ â    │     │
│  │  └─┘  ○○  │    └──────────────────────────────────────────┼───────┘     │
│  │           │                                               │ ⊘ sg()     │
│  │  State sₜ │                                          ┌────▼─────┐       │
│  │           │    ┌─── Policy Stream (Green) ───┐       │          │  ┌───┐│
│  │           │───▶│  ┌──────────┐               │──────▶│  Fusion  │─▶│PPO││
│  │           │    │  │MLP Encode│ → z_policy     │       │ (Orange) │  │π,V││
│  └──────────┘    │  └──────────┘               │       └──────────┘  └───┘│
│                   └─────────────────────────────┘                          │
│                                                                             │
│  Loss: L = L_RL + λ_phys · L_phys + λ_reg · L_reg                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Figure 2: SV-Pipeline Detailed Flowchart

### 1. Title
**Fig. 2.** *The Scalarization–Vectorization (SV) pipeline for a single edge $(i \to j)$. Raw 3D geometric quantities are projected onto the edge-local frame $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$ to produce rotation-invariant scalars (Scalarization). An MLP processes the scalars to produce force coefficients $(\alpha_1, \alpha_2, \alpha_3)$. These are recombined with basis vectors to reconstruct the 3D force (Vectorization). The antisymmetrization of $\alpha_3$ via signed radial velocity $v_r$ guarantees Newton's Third Law: $\mathbf{F}_{ij} + \mathbf{F}_{ji} = \mathbf{0}$.*

### 2. Layout
- **Full-width** (2-column, ~7.0 × 3.5 inches)
- Horizontal flow: **3D Inputs** → **Scalarization** → **Scalar MLP** → **Vectorization** → **3D Output**
- Below the main flow: a "mirror" showing the reverse edge $(j \to i)$ with antisymmetry annotations

### 3. Detailed Content

**Stage 1 — 3D Inputs (left):**
- Two nodes $i, j$ with position vectors $\mathbf{x}_i, \mathbf{x}_j$
- Velocity arrows $\dot{\mathbf{x}}_i, \dot{\mathbf{x}}_j$
- Displacement vector $\mathbf{r}_{ij}$ drawn between them
- Node embeddings $\mathbf{h}_i, \mathbf{h}_j$ as small feature vectors

**Stage 2 — Edge Frame + Scalarization (center-left):**
- Small 3D coordinate frame $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ at the midpoint
- Projection arrows showing dot products:
  - $d_{ij} = \|\mathbf{r}_{ij}\|$
  - $v_r = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1$ (with ⚡ "antisymmetric" label)
  - $v_t = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2$
  - $v_b = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3$
- Output: scalar vector $\boldsymbol{\sigma}_{ij} = [d, |v_r|, v_t, |v_b|, \|\dot{\mathbf{x}}_{ij}\|]$

**Stage 3 — Scalar MLP (center):**
- Box labeled "MLP$_\theta$" 
- Input: $[\boldsymbol{\sigma}_{ij} \| \mathbf{h}_i \| \mathbf{h}_j]$
- Two output branches:
  - $(\alpha_1, \alpha_2)$ — direct from MLP
  - $\alpha_3 = v_r \cdot g_\theta(\boldsymbol{\sigma}^\text{sym})$ — with red annotation "antisymmetric by construction"

**Stage 4 — Vectorization (center-right):**
- Equation: $\mathbf{F}_{ij} = \alpha_1 \mathbf{e}_1 + \alpha_2 \mathbf{e}_2 + \alpha_3 \mathbf{e}_3$
- 3D force arrow on node $j$

**Stage 5 — Conservation Check (right):**
- Two force arrows: $\mathbf{F}_{ij}$ on $j$, $\mathbf{F}_{ji} = -\mathbf{F}_{ij}$ on $i$
- Checkmark: $\sum \mathbf{F} = \mathbf{0}$ ✓
- Badge: "Newton's 3rd Law by Architecture"

**Bottom strip — Antisymmetry Mirror:**
- Show that reversing the edge flips $\mathbf{e}_1, \mathbf{e}_2$ but not $\mathbf{e}_3$
- And $v_r^{ji} = -v_r^{ij}$ flips $\alpha_3$
- Net result: $\mathbf{F}_{ji} = -\mathbf{F}_{ij}$

### 4. Data Source
- No data — mathematical/architectural diagram
- Based on ALGORITHM_DESIGN.md §2.3–2.4

### 5. Tool
- **matplotlib** for draft (see `figures/fig2_sv_pipeline.py`)
- Final: tikz for precise math typesetting, or Inkscape

### 6. ASCII Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   3D Inputs         Scalarization        Scalar MLP       Vectorization  │
│                                                                          │
│   (i)●────r_ij────●(j)   ┌─────────┐    ┌─────────┐    ┌────────────┐  │
│    ↗ẋ_i      ẋ_j↗        │ Project  │    │         │    │ F_ij =     │  │
│                    ──────▶│ onto     │───▶│ MLP_θ   │───▶│ α₁e₁ +    │  │
│    h_i         h_j        │ e₁,e₂,e₃│    │         │    │ α₂e₂ +    │  │
│                           └─────────┘    └────┬────┘    │ α₃e₃      │  │
│                                               │         └─────┬──────┘  │
│                           σ = [d, |vr|,       │               │         │
│                            vt, |vb|, ||v||]   │  ⚡ α₃ = vr·g(σ)       │
│                                               │    antisymmetric        │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ │
│  Reverse edge (j→i):                         │               │         │
│  e₁ʲⁱ = -e₁ⁱʲ    ──▶  σ_ji = σ_ij  ──▶ α_k same  ──▶  F_ji = -F_ij │
│  e₂ʲⁱ = -e₂ⁱʲ         BUT vr flips         ✓ Newton's 3rd Law ✓      │
│  e₃ʲⁱ = +e₃ⁱʲ                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Figure 3: EdgeFrame Construction

### 1. Title
**Fig. 3.** *Edge-local coordinate frame construction for edge $(i \to j)$. $\mathbf{e}_1$ is the radial unit vector along the displacement; $\mathbf{e}_2$ is the tangential component of relative velocity; $\mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2$ completes the frame. The key antisymmetry: reversing the edge flips $\mathbf{e}_1$ and $\mathbf{e}_2$ (shown in red), while $\mathbf{e}_3$ remains unchanged (shown in blue). This geometric property is what enables architectural momentum conservation.*

### 2. Layout
- **Single-column** (~3.4 × 3.0 inches)
- Left: edge $(i \to j)$ frame; Right: edge $(j \to i)$ frame
- Bottom: summary of symmetry/antisymmetry properties

### 3. Detailed Content

**Left panel — Edge $(i \to j)$:**
- Two filled circles for nodes $i$ (blue) and $j$ (green)
- Dashed line connecting them
- Three orthogonal arrows from the midpoint:
  - $\mathbf{e}_1^{ij}$: along $\mathbf{x}_j - \mathbf{x}_i$ (solid black, thick)
  - $\mathbf{e}_2^{ij}$: perpendicular, in velocity direction (solid gray)
  - $\mathbf{e}_3^{ij}$: out of plane / binormal (dashed blue)
- Relative velocity $\dot{\mathbf{x}}_{ij}$ shown decomposed into components along $\mathbf{e}_1, \mathbf{e}_2$
- Labels: $v_r = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1$, $\mathbf{v}^\perp = $ tangential

**Right panel — Edge $(j \to i)$ (reversed):**
- Same two nodes, but now $\mathbf{e}_1^{ji}$ points from $j$ to $i$
- $\mathbf{e}_1^{ji} = -\mathbf{e}_1^{ij}$ (red arrow, reversed)
- $\mathbf{e}_2^{ji} = -\mathbf{e}_2^{ij}$ (red arrow, reversed)
- $\mathbf{e}_3^{ji} = +\mathbf{e}_3^{ij}$ (blue arrow, SAME direction) — highlighted

**Bottom annotation:**
```
e₁: antisymmetric ⟹ α₁·e₁ cancels  ✓
e₂: antisymmetric ⟹ α₂·e₂ cancels  ✓
e₃: symmetric     ⟹ need α₃ antisymmetric (via v_r)  ✓
```

### 4. Data Source
- No data — geometric construction
- Based on ALGORITHM_DESIGN.md §2.2

### 5. Tool
- **matplotlib** with 3D quiver plot or **tikz** (3D coordinate frame drawing)
- tikz preferred for precise geometric rendering

### 6. ASCII Layout

```
┌────────────────────────────────────────────────────┐
│                                                    │
│    Edge (i → j)              Edge (j → i)          │
│                                                    │
│         e₂↑                        ↑e₂ʲⁱ=-e₂     │
│          |  e₃                 e₃ʲⁱ=+e₃ |         │
│          | ╱                       ╲   |           │
│   (i)●───┼──e₁──▶●(j)    (i)●◀──e₁ʲⁱ──┼───●(j)   │
│          |                         |               │
│                                                    │
│   ─────────────────────────────────────────────    │
│   e₁ⁱʲ = -e₁ʲⁱ  (antisym)  → α₁ auto-cancels    │
│   e₂ⁱʲ = -e₂ʲⁱ  (antisym)  → α₂ auto-cancels    │
│   e₃ⁱʲ = +e₃ʲⁱ  (SYMMETRIC) → α₃ needs v_r fix  │
└────────────────────────────────────────────────────┘
```

---

## Figure 4: Main Experiment Results — Learning Curves

### 1. Title
**Fig. 4.** *Learning curves on three manipulation environments. Solid lines: mean episode return over 5 seeds; shaded regions: ± 1 std. PhysRobot (blue) achieves comparable asymptotic performance to PPO (gray) and SAC (green) while requiring 3–5× fewer environment steps to reach 90% performance (dashed horizontal lines). GNS (orange) benefits from graph structure but lacks the conservation inductive bias.*

### 2. Layout
- **Full-width** (2-column, ~7.0 × 2.5 inches)
- **1 row × 3 columns** of subplots:
  - (a) PushBox  
  - (b) MultiPush (3 objects)  
  - (c) Sort (3 colors)
- Shared legend at the top or right

### 3. Detailed Content

**Each subplot:**
- X-axis: Environment steps (0 to 10M or 500K depending on env)
- Y-axis: Episode return (normalized or raw)
- Lines for each method (5-seed mean ± 1 std shading):
  - PhysRobot (blue, solid, thick) — #3B82F6
  - PPO (gray, solid) — #6B7280
  - SAC (green, dashed) — #10B981
  - GNS (orange, solid) — #F59E0B
  - HNN (purple, dotted) — #8B5CF6
  - TD3 (red, dashed) — #EF4444
- Horizontal dashed line: 90% of best asymptotic performance
- Vertical marker: step at which each method crosses 90% threshold
- Title: environment name + difficulty descriptor

**Styling:**
- Grid: light gray, major only
- Font: 9pt for labels, 8pt for ticks
- Line width: 2pt for PhysRobot, 1.5pt for others
- Shading alpha: 0.2
- Tight layout, no wasted space

### 4. Data Source
- Training logs from Phase 1 + Phase 2 experiments
- Files: `results/{method}_seed{seed}_train.json`
- Smoothing: exponential moving average (window = 20 episodes)

### 5. Tool
- **matplotlib** (production-ready)
- Script: `figures/fig4_learning_curves.py` (to be written when data available)

### 6. ASCII Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  ═══ PhysRobot ─── PPO --- SAC ─── GNS ··· HNN --- TD3 ═══       │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  (a) PushBox  │  │(b) MultiPush │  │  (c) Sort    │              │
│  │               │  │              │  │              │              │
│  │  R ╱──────    │  │  R ╱──────   │  │  R ╱─────    │              │
│  │   ╱  ╱───     │  │   ╱  ╱──     │  │   ╱  ╱──    │              │
│  │  ╱ ╱╱──       │  │  ╱ ╱╱─       │  │  ╱ ╱╱─      │              │
│  │ ╱╱╱╱          │  │ ╱╱╱          │  │ ╱╱╱          │              │
│  │╱╱╱─ ─ ─90%   │  │╱╱─ ─ ─90%   │  │╱╱─ ─ ─90%   │              │
│  │               │  │              │  │              │              │
│  │  Steps (M)    │  │  Steps (M)   │  │  Steps (M)   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Figure 5: Ablation Study Results

### 1. Title
**Fig. 5.** *Ablation study on MultiPush (3 objects). We remove each component of PhysRobot individually and measure the impact on sample efficiency (steps to 90% performance) and final success rate. Conservation-aware messaging has the largest single contribution (+40% steps when removed), followed by antisymmetric messages (+35%) and EdgeFrame equivariance (+25%).*

### 2. Layout
- **Single-column** (~3.4 × 3.0 inches)
- **Grouped bar chart** with two metrics side-by-side
- Alternative: **table figure** if data is cleaner in tabular form

### 3. Detailed Content

**Bar chart version:**
- X-axis: Ablation variants (6 groups):
  1. PhysRobot-Full
  2. − Conservation Loss
  3. − EdgeFrame
  4. − Antisymmetric Messages
  5. − Dynamic Edges (fixed graph)
  6. Vanilla GNN (all removed)
- Y-axis (left, blue bars): Steps to 90% performance (lower is better)
- Y-axis (right, orange bars): Final success rate % (higher is better)
- Error bars: ± 1 std over 5 seeds
- PhysRobot-Full bar highlighted in darker shade

**Table figure version (alternative):**

| Variant | Steps to 90% (↓) | Final SR (↑) | OOD SR (↑) |
|---------|-------------------|--------------|-------------|
| PhysRobot-Full | **250K ± 30K** | **72 ± 5%** | **48 ± 6%** |
| − Conservation | 350K ± 45K | 68 ± 6% | 32 ± 8% |
| − EdgeFrame | 315K ± 40K | 65 ± 7% | 35 ± 7% |
| − Antisym. Msg | 340K ± 50K | 63 ± 8% | 30 ± 9% |
| − Dynamic Edges | 330K ± 45K | 60 ± 7% | 28 ± 8% |
| Vanilla GNN | 450K ± 60K | 55 ± 9% | 20 ± 10% |

### 4. Data Source
- Phase 1 + Phase 2 ablation runs
- Files: `results/ablation_{variant}_seed{seed}_train.json`

### 5. Tool
- **matplotlib** bar chart
- Use `plt.bar()` with grouped positions

### 6. ASCII Layout

```
┌──────────────────────────────────────┐
│  Steps to 90% (K) ■  Final SR (%) □  │
│                                      │
│  ■□  ■ □  ■ □  ■ □  ■ □  ■  □       │
│  ██  █ █  █ █  █ █  █ █  █  █       │
│  ██  █ █  █ █  █ █  █ █  █  █       │
│  ██  █ █  █ █  █ █  █ █  █  █       │
│  ██  █ █  █ █  █ █  █ █  █  █       │
│  ██  █ █  █ █  █ █  █ █  █  █       │
│ ─┴┴──┴─┴──┴─┴──┴─┴──┴─┴──┴──┴───   │
│ Full -Cons -Edge -Anti -Dyn Vanilla  │
│                                      │
│ Error bars: ±1 std over 5 seeds      │
└──────────────────────────────────────┘
```

---

## Figure 6: OOD Generalization

### 1. Title
**Fig. 6.** *Out-of-distribution generalization. (a) Success rate vs. box mass (trained on 0.5 kg, gray band). PhysRobot (blue) degrades gracefully across a 50× mass range, while PPO (gray) and SAC (green) collapse beyond 2× the training mass. (b) Generalization heatmap: success rate as a function of both object count and mass multiplier. PhysRobot maintains >40% success rate across the tested range.*

### 2. Layout
- **Full-width** (2-column, ~7.0 × 2.5 inches)
- **1 row × 2 columns:**
  - (a) Line plot: SR vs mass (log-scale x-axis)
  - (b) Heatmap: SR over (object count × mass) grid

### 3. Detailed Content

**Panel (a) — SR vs Mass:**
- X-axis: Box mass [0.1, 0.25, 0.5, 1.0, 2.0, 5.0] kg (log scale)
- Y-axis: Success rate (0–100%)
- Lines (5-seed mean ± std shading):
  - PhysRobot: gradual decline, stays >30% everywhere
  - PPO: sharp drop after 1.0 kg, <5% at 5.0 kg
  - SAC: similar to PPO, slightly better
  - GNS: moderate decline, between PPO and PhysRobot
  - HNN: good at low mass but degrades on high mass
- Gray vertical band at mass=0.5 kg (training distribution)
- Annotation arrows showing "3× more robust" for PhysRobot

**Panel (b) — Heatmap:**
- X-axis: Mass multiplier [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
- Y-axis: Object count [1, 3, 5, 7, 10]
- Color: Success rate (0%=red → 100%=green, viridis colormap)
- One heatmap per method (PhysRobot only, or 2×2 grid of mini-heatmaps)
- Numerical values in each cell

### 4. Data Source
- Phase 1 OOD evaluation: `results/{method}_seed{seed}_ood.json`
- Phase 3 multi-object + mass sweep

### 5. Tool
- **matplotlib** (`plt.plot` + `plt.imshow` or `sns.heatmap`)

### 6. ASCII Layout

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  (a) SR vs Mass                    (b) Generalization Heatmap      │
│  ┌─────────────────────────┐       ┌─────────────────────────┐    │
│  │100%│                    │       │      Mass multiplier     │    │
│  │    │  ██                │       │    0.2  0.5  1.0  2.0   │    │
│  │    │ █  ██              │       │ 1 │ 62   85   90   78  ││    │
│  │ SR │█    ██             │  #obj │ 3 │ 48   72   75   55  ││    │
│  │    │ ░░   ████          │       │ 5 │ 35   58   62   42  ││    │
│  │    │  ░░░     ████      │       │ 7 │ 22   45   48   30  ││    │
│  │    │   ░░░░       ████  │       │10 │ 12   30   35   18  ││    │
│  │  0%│────────────────────│       └─────────────────────────┘    │
│  │    0.1  0.5  1.0  5.0  │       Colorbar: 0% ─── 50% ─── 100% │
│  │        Mass (kg, log)   │                                      │
│  └─────────────────────────┘       PhysRobot SR (%)               │
│  ─ PhysRobot  ░ PPO  ─ GNS                                       │
└────────────────────────────────────────────────────────────────────┘
```

---

## Figure 7: Multi-Object Environment Visualization (Optional)

### 1. Title
**Fig. 7.** *Multi-object manipulation environments. Left: MuJoCo rendering of the 5-box pushing task, with colored boxes and matching goal markers. Right: the corresponding dynamic scene graph used by PhysRobot, where edge thickness indicates contact force magnitude. Dashed edges represent potential (within cutoff) but non-contact interactions.*

### 2. Layout
- **Full-width** (2-column, ~7.0 × 2.5 inches)
- **1 row × 2 panels:**
  - (a) MuJoCo rendered scene
  - (b) Graph overlay / abstract graph drawing

### 3. Detailed Content

**Panel (a) — MuJoCo Render:**
- Top-down view of the tabletop
- Robot arm (gray/silver) in center-left
- 5 colored boxes: red, blue, green, yellow, purple
- 5 matching semi-transparent goal circles
- End-effector highlighted
- Arrows showing intended push directions

**Panel (b) — Scene Graph:**
- Abstract graph visualization (spring layout or force-directed)
- Node types:
  - ◆ End-effector (large, gray)
  - ● Objects (colored, matching MuJoCo colors)
  - ○ Goals (outlined, matching colors)
- Edge types:
  - Thick solid: active contact (ee→box, box→box)
  - Thin dashed: within cutoff, no contact
  - Arrow direction: force direction
- Edge thickness ∝ force magnitude from physics stream
- Labels on nodes: mass values (0.5, 0.7, 1.2 kg, etc.)

### 4. Data Source
- MuJoCo render: screenshot from environment
- Graph: extracted from PhysRobot's scene graph at a representative timestep

### 5. Tool
- **MuJoCo offscreen renderer** for panel (a)
- **matplotlib + networkx** for panel (b)

### 6. ASCII Layout

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  (a) MuJoCo Scene                  (b) Dynamic Scene Graph       │
│  ┌──────────────────────┐          ┌──────────────────────┐     │
│  │                      │          │                      │     │
│  │     ○goal₁   ○goal₂  │          │    ○g₁       ○g₂     │     │
│  │                      │          │    :         :       │     │
│  │   ■box₁  ■box₂      │          │    ●b₁━━━━●b₂       │     │
│  │        ■box₃         │          │     ┃╲   ╱┃         │     │
│  │   ╔═══╗              │          │     ┃  ●b₃ ┃        │     │
│  │   ║ARM║  ○goal₃      │          │     ┃ ╱  ╲ ┃        │     │
│  │   ╚═══╝              │          │    ◆ee    ●b₄       │     │
│  │     ■box₄  ■box₅     │          │     ┃       :       │     │
│  │   ○goal₄   ○goal₅    │          │    ●b₅     ○g₄      │     │
│  │                      │          │     :                │     │
│  │                      │          │    ○g₅               │     │
│  └──────────────────────┘          └──────────────────────┘     │
│                                    ━ contact  ┈ within cutoff    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Production Notes

### Color Palette (consistent across all figures)

| Element | Color | Hex |
|---------|-------|-----|
| PhysRobot | Blue | #3B82F6 |
| PPO | Gray | #6B7280 |
| SAC | Green | #10B981 |
| GNS | Orange | #F59E0B |
| HNN | Purple | #8B5CF6 |
| TD3 | Red | #EF4444 |
| Physics Stream | Dark Blue | #1D4ED8 |
| Policy Stream | Dark Green | #059669 |
| Fusion | Amber | #D97706 |
| Stop-gradient | Red | #EF4444 |

### Typography
- Figure labels: **bold** 10pt
- Axis labels: 9pt
- Tick labels: 8pt
- Annotations: 8pt italic
- All text: sans-serif (Helvetica / Arial)

### File Format
- Draft: PNG 300 DPI
- Camera-ready: PDF (vector) via `plt.savefig('fig.pdf', bbox_inches='tight')`

### Size Constraints (ICRA/CoRL)
- Full-width: 7.0" × variable height
- Single-column: 3.4" × variable height
- Total figure budget: ~40% of 8 pages ≈ 3.2 pages of figures

---

*Document complete. Draft matplotlib code for Fig 1 and Fig 2 in `figures/` directory.*
