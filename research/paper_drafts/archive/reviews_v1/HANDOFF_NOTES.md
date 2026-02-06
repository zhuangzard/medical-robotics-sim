# Paper-Writer → Other Agents: Handoff Notes

**Date:** 2026-02-06  
**From:** paper-writer  
**To:** paper-algorithm, paper-experiment, paper-reviewer

---

## What's Ready

### 1. `PAPER_OUTLINE.md` — Full Paper Outline
- Title, Abstract (200 words), and complete 5-section structure
- Section 3 (Method) includes detailed math for scalarization–vectorization, Theorem 1 (momentum conservation), and the dual-stream architecture
- Section 4 (Experiments) has all table/figure templates with 4 environments, 6 baselines, and 4 ablation studies
- Writing schedule from W1–W7

### 2. `RELATED_WORK.md` — Full Related Work Draft
- 52 references across 3 main areas
- Detailed positioning table (Table 1) comparing 10 methods
- Analysis of why HNN/LNN, EGNN, differentiable physics, and foundation models are insufficient
- Key narrative and elevator pitch

---

## For `paper-algorithm`:
- **Priority task:** Implement the scalarization–vectorization pipeline (Section 3.3 of PAPER_OUTLINE.md)
- The outline specifies the exact 5-step pipeline: Graph Construction → Edge Frame → Scalarization → Vectorization → Aggregation
- Theorem 1 needs formal verification in code: test that `F_ij = -F_ji` for all learned parameters
- Verify the mathematical claims in RELATED_WORK.md Section 2.4 (positioning analysis)

## For `paper-experiment`:
- **Priority task:** Set up the 4 environments (PushBox, MultiPush, StackCubes, Rearrange)
- Tables 1–5 in PAPER_OUTLINE.md define all experiments needed
- 6 baselines to implement: PPO, SAC, GNS+PPO, EGNN+PPO, Dreamer v3, PPO+PhysLoss
- Ablation study design is in Section 4.4 (7 variants)
- OOD experiments: mass ×{0.5, 2.0, 5.0}, friction ×{0.5, 2.0}
- All experiments need 5 seeds with mean ± std

## For `paper-reviewer`:
- **Priority task:** Review both documents for:
  - Structural completeness (are we missing any section?)
  - Claim validity (are any claims too strong without evidence?)
  - Missing related work (especially 2025-2026 papers)
  - Potential reviewer attacks (the feasibility report Section 4.2 lists 6 anticipated questions)
  - The "hard constraint vs soft constraint" narrative — is this defensible?

---

*paper-writer agent, 2026-02-06*
