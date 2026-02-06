#!/usr/bin/env python3
"""
Fig 2: SV-Pipeline Detailed Flowchart — Draft Skeleton
Visualizes the Scalarization–Vectorization pipeline for momentum-conserving
message passing.

Usage:
    python fig2_sv_pipeline.py          # saves fig2_sv_pipeline.png
    python fig2_sv_pipeline.py --pdf    # saves fig2_sv_pipeline.pdf
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

# ── Color palette ──────────────────────────────────────────────────────────
BLUE       = "#3B82F6"
BLUE_DARK  = "#1D4ED8"
BLUE_LIGHT = "#DBEAFE"
GREEN      = "#10B981"
GREEN_DARK = "#059669"
GREEN_LIGHT= "#D1FAE5"
ORANGE     = "#F59E0B"
ORANGE_LIGHT="#FEF3C7"
RED        = "#EF4444"
RED_LIGHT  = "#FEE2E2"
PURPLE     = "#8B5CF6"
PURPLE_LIGHT="#EDE9FE"
GRAY       = "#6B7280"
GRAY_LIGHT = "#F3F4F6"
DARK       = "#1F2937"
WHITE      = "#FFFFFF"

# ── Figure setup ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=200)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.set_aspect("equal")
ax.axis("off")


def draw_box(ax, xy, w, h, label, color, text_color=DARK, fontsize=7.5,
             sublabel=None, linewidth=1.2, alpha=0.9):
    """Draw a rounded rectangle with label."""
    box = FancyBboxPatch(xy, w, h,
                         boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor=DARK,
                         linewidth=linewidth, alpha=alpha, zorder=2)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    if sublabel:
        ax.text(cx, cy + 0.18, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=3)
        ax.text(cx, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=text_color, zorder=3, style="italic")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=3)


def arrow(ax, start, end, color=DARK, lw=1.5, style="-|>", cs="arc3,rad=0"):
    a = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                        linewidth=lw, mutation_scale=12,
                        connectionstyle=cs, zorder=4)
    ax.add_patch(a)


def stage_label(ax, x, y, text, color=DARK):
    """Draw a stage header label."""
    ax.text(x, y, text, ha="center", va="center",
            fontsize=8, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=WHITE,
                      edgecolor=color, linewidth=1.2),
            zorder=5)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE HEADERS (top row)
# ═══════════════════════════════════════════════════════════════════════════

headers = [
    (1.5,  7.5, "① 3D Inputs",      BLUE_DARK),
    (4.5,  7.5, "② Edge Frame",     GREEN_DARK),
    (7.2,  7.5, "③ Scalarization",  ORANGE),
    (9.8,  7.5, "④ Scalar MLP",     PURPLE),
    (12.5, 7.5, "⑤ Vectorization",  RED),
]
for x, y, t, c in headers:
    stage_label(ax, x, y, t, c)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: 3D INPUTS (left)
# ═══════════════════════════════════════════════════════════════════════════

# Nodes i and j
node_i = (0.8, 5.5)
node_j = (2.2, 5.5)

ax.plot(*node_i, "o", color=BLUE, markersize=14, zorder=5)
ax.plot(*node_j, "o", color=GREEN, markersize=14, zorder=5)
ax.text(node_i[0], node_i[1], "i", ha="center", va="center",
        fontsize=8, color=WHITE, fontweight="bold", zorder=6)
ax.text(node_j[0], node_j[1], "j", ha="center", va="center",
        fontsize=8, color=WHITE, fontweight="bold", zorder=6)

# Displacement vector
arrow(ax, (0.95, 5.5), (2.05, 5.5), color=DARK, lw=2)
ax.text(1.5, 5.8, r"$\mathbf{r}_{ij}$", ha="center", fontsize=7, color=DARK)

# Velocity arrows
ax.annotate("", xy=(1.2, 6.1), xytext=node_i,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
ax.text(0.6, 6.0, r"$\dot{\mathbf{x}}_i$", fontsize=6, color=BLUE)

ax.annotate("", xy=(2.6, 6.2), xytext=node_j,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
ax.text(2.7, 6.0, r"$\dot{\mathbf{x}}_j$", fontsize=6, color=GREEN)

# Node embeddings
draw_box(ax, (0.3, 4.3), 0.9, 0.6, r"$\mathbf{h}_i$", BLUE_LIGHT, fontsize=6.5)
draw_box(ax, (1.8, 4.3), 0.9, 0.6, r"$\mathbf{h}_j$", GREEN_LIGHT, fontsize=6.5)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: EDGE FRAME CONSTRUCTION (center-left)
# ═══════════════════════════════════════════════════════════════════════════

# Edge frame origin
frame_o = (4.5, 5.5)

# Draw coordinate frame axes
# e1 (radial) — along displacement
ax.annotate("", xy=(5.5, 5.5), xytext=frame_o,
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
ax.text(5.6, 5.3, r"$\mathbf{e}_1$", fontsize=7, color=RED, fontweight="bold")

# e2 (tangential) — perpendicular in plane
ax.annotate("", xy=(4.5, 6.6), xytext=frame_o,
            arrowprops=dict(arrowstyle="-|>", color=GREEN_DARK, lw=2))
ax.text(4.1, 6.6, r"$\mathbf{e}_2$", fontsize=7, color=GREEN_DARK, fontweight="bold")

# e3 (binormal) — out of plane (shown at angle)
ax.annotate("", xy=(3.8, 6.2), xytext=frame_o,
            arrowprops=dict(arrowstyle="-|>", color=BLUE_DARK, lw=2,
                            linestyle="dashed"))
ax.text(3.4, 6.3, r"$\mathbf{e}_3$", fontsize=7, color=BLUE_DARK, fontweight="bold")

# Frame construction equations
frame_eqs = [
    r"$\mathbf{e}_1 = \frac{\mathbf{r}_{ij}}{\|\mathbf{r}_{ij}\|}$",
    r"$\mathbf{e}_2 = \frac{\mathbf{v}^\perp}{\|\mathbf{v}^\perp\|}$",
    r"$\mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2$",
]
for k, eq in enumerate(frame_eqs):
    ax.text(4.5, 4.5 - k * 0.4, eq, ha="center", fontsize=6, color=DARK)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: SCALARIZATION (center)
# ═══════════════════════════════════════════════════════════════════════════

draw_box(ax, (6.2, 4.8), 2.0, 1.8, "Scalarize", ORANGE_LIGHT,
         sublabel="(project onto frame)", fontsize=7)

# Scalar features listed
scalars = [
    r"$d_{ij} = \|\mathbf{r}_{ij}\|$",
    r"$v_r = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1$" + "  ⚡",
    r"$v_t = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2$",
    r"$v_b = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3$",
    r"$\|\dot{\mathbf{x}}_{ij}\|$",
]
for k, s in enumerate(scalars):
    ax.text(7.2, 4.55 - k * 0.35, s, ha="center", fontsize=5.5, color=DARK)

# Antisymmetric marker on v_r
ax.text(8.5, 4.2, "antisym!", ha="center", fontsize=5,
        color=RED, fontweight="bold", style="italic")

# Output arrow
arrow(ax, (6.0, 5.5), (6.2, 5.5), color=DARK)

# Extended features annotation
ax.text(7.2, 3.3,
        r"$\boldsymbol{\sigma}^{\mathrm{ext}} = [\boldsymbol{\sigma} \| \mathbf{h}_i \| \mathbf{h}_j]$",
        ha="center", fontsize=6.5, color=DARK,
        bbox=dict(boxstyle="round,pad=0.15", facecolor=GRAY_LIGHT,
                  edgecolor=GRAY, linewidth=0.8))


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: SCALAR MLP (center-right)
# ═══════════════════════════════════════════════════════════════════════════

draw_box(ax, (8.8, 4.8), 2.0, 1.8, r"MLP$_\theta$", PURPLE_LIGHT,
         sublabel="(rotation-invariant)", fontsize=7)

# Two output branches
# α₁, α₂ branch
ax.text(9.8, 4.5, r"$(\alpha_1, \alpha_2)$", ha="center", fontsize=6.5,
        color=PURPLE, fontweight="bold")
ax.text(9.8, 4.15, "symmetric", ha="center", fontsize=5, color=PURPLE, style="italic")

# α₃ branch (highlighted — antisymmetrized)
draw_box(ax, (8.9, 2.6), 1.8, 1.0, "", RED_LIGHT, linewidth=1.5)
ax.text(9.8, 3.3, r"$\alpha_3 = v_r \cdot g_\theta(\boldsymbol{\sigma}^{\mathrm{sym}})$",
        ha="center", fontsize=6, color=RED, fontweight="bold")
ax.text(9.8, 2.85, "← antisymmetric\n   by construction", ha="center",
        fontsize=5, color=RED, style="italic")

# Arrows
arrow(ax, (8.2, 5.7), (8.8, 5.7), color=DARK)
arrow(ax, (9.8, 4.8), (9.8, 3.6), color=RED, lw=1.2)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5: VECTORIZATION (right)
# ═══════════════════════════════════════════════════════════════════════════

draw_box(ax, (11.2, 4.8), 2.5, 1.8, "Vectorize", BLUE_LIGHT, fontsize=7,
         sublabel="(reconstruct 3D)")

# Force equation
ax.text(12.45, 4.5,
        r"$\mathbf{F}_{ij} = \alpha_1\mathbf{e}_1 + \alpha_2\mathbf{e}_2 + \alpha_3\mathbf{e}_3$",
        ha="center", fontsize=6, color=DARK, fontweight="bold")

# Arrow from MLP to vectorization
arrow(ax, (10.8, 5.7), (11.2, 5.7), color=DARK)

# 3D force arrow output
ax.annotate("", xy=(13.5, 6.8), xytext=(12.45, 6.6),
            arrowprops=dict(arrowstyle="-|>", color=BLUE_DARK, lw=3))
ax.text(13.2, 7.1, r"$\mathbf{F}_{ij}$", ha="center", fontsize=8,
        color=BLUE_DARK, fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════════
# BOTTOM: ANTISYMMETRY MIRROR / CONSERVATION PROOF
# ═══════════════════════════════════════════════════════════════════════════

# Separator line
ax.plot([0.3, 13.7], [2.2, 2.2], color=GRAY, linewidth=1, linestyle="--", zorder=1)
ax.text(0.5, 2.35, "Conservation Proof (reverse edge j→i)", ha="left",
        fontsize=7, fontweight="bold", color=DARK)

# Three columns showing antisymmetry
cols = [
    (2.0, "Frame Symmetry",
     [r"$\mathbf{e}_1^{ji} = -\mathbf{e}_1^{ij}$",
      r"$\mathbf{e}_2^{ji} = -\mathbf{e}_2^{ij}$",
      r"$\mathbf{e}_3^{ji} = +\mathbf{e}_3^{ij}$"]),
    (6.0, "Coefficient Symmetry",
     [r"$\alpha_1^{ji} = \alpha_1^{ij}$  (same input)",
      r"$\alpha_2^{ji} = \alpha_2^{ij}$  (same input)",
      r"$\alpha_3^{ji} = -\alpha_3^{ij}$  ($v_r$ flips!)"]),
    (11.0, "Cancellation",
     [r"$\alpha_1 \mathbf{e}_1$: cancels  ✓",
      r"$\alpha_2 \mathbf{e}_2$: cancels  ✓",
      r"$\alpha_3 \mathbf{e}_3$: cancels  ✓"]),
]

for cx, title, lines in cols:
    ax.text(cx, 1.9, title, ha="center", fontsize=6.5, fontweight="bold",
            color=BLUE_DARK)
    for k, line in enumerate(lines):
        # Color the e3 line differently
        c = RED if "e_3" in line or "v_r" in line else DARK
        ax.text(cx, 1.5 - k * 0.35, line, ha="center", fontsize=5.5, color=c)

# Final result box
draw_box(ax, (4.5, 0.2), 5.0, 0.6, "", GREEN_LIGHT, linewidth=2)
ax.text(7.0, 0.5,
        r"$\mathbf{F}_{ij} + \mathbf{F}_{ji} = \mathbf{0}$  →  "
        r"$\sum_i \mathbf{F}_i = \mathbf{0}$  ✓  "
        "Newton's 3rd Law by Architecture",
        ha="center", va="center", fontsize=7, color=GREEN_DARK, fontweight="bold",
        zorder=5)


# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════

plt.tight_layout(pad=0.2)

if "--pdf" in sys.argv:
    plt.savefig("fig2_sv_pipeline.pdf", bbox_inches="tight", dpi=300)
    print("Saved fig2_sv_pipeline.pdf")
else:
    plt.savefig("fig2_sv_pipeline.png", bbox_inches="tight", dpi=300)
    print("Saved fig2_sv_pipeline.png")

plt.close()
