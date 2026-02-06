#!/usr/bin/env python3
"""
Fig 1: PhysRobot System Overview — Draft Skeleton
Generates a block-diagram overview of the dual-stream architecture.

Usage:
    python fig1_system_overview.py          # saves fig1_system_overview.png
    python fig1_system_overview.py --pdf    # saves fig1_system_overview.pdf
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette ──────────────────────────────────────────────────────────
BLUE_LIGHT = "#93C5FD"
BLUE       = "#3B82F6"
BLUE_DARK  = "#1D4ED8"
GREEN_LIGHT= "#A7F3D0"
GREEN      = "#10B981"
GREEN_DARK = "#059669"
ORANGE_LIGHT="#FDE68A"
ORANGE     = "#F59E0B"
ORANGE_DARK= "#D97706"
RED        = "#EF4444"
GRAY       = "#6B7280"
GRAY_LIGHT = "#E5E7EB"
WHITE      = "#FFFFFF"
DARK       = "#1F2937"

# ── Figure setup ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.5), dpi=200)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.set_aspect("equal")
ax.axis("off")


def draw_box(ax, xy, w, h, label, color, text_color=DARK, fontsize=8,
             sublabel=None, rounded=True, linewidth=1.5, alpha=0.85):
    """Draw a rounded rectangle with label."""
    style = "round,pad=0.1" if rounded else "square,pad=0.05"
    box = FancyBboxPatch(xy, w, h,
                         boxstyle=style,
                         facecolor=color, edgecolor=DARK,
                         linewidth=linewidth, alpha=alpha,
                         zorder=2)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    if sublabel:
        ax.text(cx, cy + 0.15, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=3)
        ax.text(cx, cy - 0.25, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=text_color, zorder=3, style="italic")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=3)


def arrow(ax, start, end, color=DARK, lw=1.5, style="-|>", connectionstyle="arc3,rad=0"):
    """Draw an arrow."""
    a = FancyArrowPatch(start, end,
                        arrowstyle=style, color=color,
                        linewidth=lw, mutation_scale=12,
                        connectionstyle=connectionstyle,
                        zorder=4)
    ax.add_patch(a)


# ═══════════════════════════════════════════════════════════════════════════
# INPUT BLOCK (left)
# ═══════════════════════════════════════════════════════════════════════════

# MuJoCo scene box
draw_box(ax, (0.3, 1.5), 2.4, 4.0, "", GRAY_LIGHT, linewidth=2)
ax.text(1.5, 5.15, "MuJoCo Environment", ha="center", va="center",
        fontsize=7, fontweight="bold", color=DARK)

# Robot arm (simple L-shape)
ax.plot([1.0, 1.0, 1.8], [2.3, 3.5, 3.5], color=GRAY, linewidth=4,
        solid_capstyle="round", zorder=3)
ax.plot(1.0, 2.3, "o", color=GRAY, markersize=8, zorder=3)  # base
ax.plot(1.8, 3.5, "s", color=BLUE_DARK, markersize=6, zorder=3)  # ee

# Boxes
for (bx, by, c) in [(1.6, 2.3, "#EF4444"), (2.0, 2.7, "#3B82F6")]:
    rect = plt.Rectangle((bx, by), 0.3, 0.3, facecolor=c, edgecolor=DARK,
                          linewidth=1, zorder=3, alpha=0.8)
    ax.add_patch(rect)

# Goals (circles)
for (gx, gy, c) in [(2.2, 4.0, "#EF4444"), (1.2, 4.3, "#3B82F6")]:
    circle = plt.Circle((gx, gy), 0.12, facecolor=c, edgecolor=DARK,
                         linewidth=0.8, alpha=0.3, zorder=3)
    ax.add_patch(circle)

# State output label
ax.text(1.5, 1.7, r"State $\mathbf{s}_t$", ha="center", va="center",
        fontsize=7, color=DARK, style="italic")


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS STREAM (top-center) — Blue
# ═══════════════════════════════════════════════════════════════════════════

# Background band
physics_bg = FancyBboxPatch((3.2, 3.8), 6.6, 2.6,
                             boxstyle="round,pad=0.15",
                             facecolor=BLUE_LIGHT, edgecolor=BLUE_DARK,
                             linewidth=1.5, alpha=0.3, zorder=1)
ax.add_patch(physics_bg)
ax.text(3.6, 6.15, "Physics Stream", ha="left", va="center",
        fontsize=8, fontweight="bold", color=BLUE_DARK)

# Scene Graph Construction
draw_box(ax, (3.5, 4.2), 1.8, 1.2, "Scene Graph", BLUE_LIGHT,
         sublabel="Construction", fontsize=7)

# Small graph icon inside
for (nx, ny) in [(3.9, 4.7), (4.6, 5.1), (4.9, 4.5)]:
    ax.plot(nx, ny, "o", color=BLUE_DARK, markersize=4, zorder=5)
ax.plot([3.9, 4.6], [4.7, 5.1], color=BLUE_DARK, linewidth=0.8, zorder=4)
ax.plot([4.6, 4.9], [5.1, 4.5], color=BLUE_DARK, linewidth=0.8, zorder=4)
ax.plot([3.9, 4.9], [4.7, 4.5], color=BLUE_DARK, linewidth=0.8, zorder=4)

# SV Message Passing
draw_box(ax, (5.8, 4.2), 1.8, 1.2, "SV Message", BLUE,
         sublabel="Passing (×L)", text_color=WHITE, fontsize=7)
# Conservation badge
ax.text(7.4, 5.2, "✓ N3L", ha="center", va="center",
        fontsize=5.5, color=GREEN_DARK,
        bbox=dict(boxstyle="round,pad=0.15", facecolor=GREEN_LIGHT,
                  edgecolor=GREEN_DARK, linewidth=0.8),
        zorder=5)

# Dynamics Decoder
draw_box(ax, (8.1, 4.2), 1.5, 1.2, "Dynamics", BLUE_DARK,
         sublabel=r"Decoder → $\hat{\mathbf{a}}$", text_color=WHITE, fontsize=7)

# Arrows in Physics Stream
arrow(ax, (5.3, 4.8), (5.8, 4.8), color=BLUE_DARK)
arrow(ax, (7.6, 4.8), (8.1, 4.8), color=BLUE_DARK)

# Physics loss (self-supervised) — side annotation
ax.annotate(r"$\mathcal{L}_\mathrm{phys}$" + "\n(self-supervised)",
            xy=(8.85, 4.2), xytext=(8.85, 3.5),
            ha="center", va="top", fontsize=6, color=BLUE_DARK,
            arrowprops=dict(arrowstyle="->", color=BLUE_DARK, lw=1),
            zorder=5)


# ═══════════════════════════════════════════════════════════════════════════
# POLICY STREAM (bottom-center) — Green
# ═══════════════════════════════════════════════════════════════════════════

policy_bg = FancyBboxPatch((3.2, 0.8), 4.5, 2.0,
                            boxstyle="round,pad=0.15",
                            facecolor=GREEN_LIGHT, edgecolor=GREEN_DARK,
                            linewidth=1.5, alpha=0.3, zorder=1)
ax.add_patch(policy_bg)
ax.text(3.6, 2.55, "Policy Stream", ha="left", va="center",
        fontsize=8, fontweight="bold", color=GREEN_DARK)

draw_box(ax, (4.0, 1.1), 1.8, 1.2, "MLP Encoder", GREEN,
         sublabel=r"$\mathbf{z}_\mathrm{policy}$", text_color=WHITE, fontsize=7)

arrow(ax, (5.8, 1.7), (7.3, 1.7), color=GREEN_DARK)
ax.text(6.55, 1.95, r"$\mathbf{z}_\mathrm{policy}$", ha="center",
        fontsize=6.5, color=GREEN_DARK, style="italic")


# ═══════════════════════════════════════════════════════════════════════════
# SPLIT ARROWS from Input
# ═══════════════════════════════════════════════════════════════════════════

# Top arrow to Physics Stream
arrow(ax, (2.7, 4.0), (3.5, 4.8), color=DARK, lw=1.5,
      connectionstyle="arc3,rad=-0.15")
# Bottom arrow to Policy Stream
arrow(ax, (2.7, 2.5), (4.0, 1.7), color=DARK, lw=1.5,
      connectionstyle="arc3,rad=0.15")


# ═══════════════════════════════════════════════════════════════════════════
# STOP-GRADIENT + FUSION (right-center) — Orange
# ═══════════════════════════════════════════════════════════════════════════

# Stop-gradient symbol
ax.text(10.2, 4.8, "⊘", ha="center", va="center",
        fontsize=14, color=RED, fontweight="bold", zorder=5)
ax.text(10.2, 4.35, "stop-grad", ha="center", va="center",
        fontsize=5.5, color=RED, zorder=5)

# Arrow from decoder to stop-grad
arrow(ax, (9.6, 4.8), (9.9, 4.8), color=BLUE_DARK)

# Fusion box
draw_box(ax, (10.6, 2.2), 1.3, 2.6, "Fusion", ORANGE,
         text_color=WHITE, fontsize=8)

# Arrow from stop-grad down to fusion
arrow(ax, (10.2, 4.3), (11.25, 4.8 - 2.0), color=RED, lw=1.2,
      connectionstyle="arc3,rad=-0.1")
ax.text(10.5, 3.65, r"sg($\hat{\mathbf{a}}$)", ha="center",
        fontsize=6, color=RED, style="italic")

# Arrow from policy stream to fusion
arrow(ax, (7.3, 1.7), (10.6, 3.0), color=GREEN_DARK, lw=1.2,
      connectionstyle="arc3,rad=-0.2")


# ═══════════════════════════════════════════════════════════════════════════
# PPO OUTPUT (far right)
# ═══════════════════════════════════════════════════════════════════════════

# PPO Actor
draw_box(ax, (12.4, 3.6), 1.2, 1.0, "PPO Actor", ORANGE_DARK,
         sublabel=r"$\pi(a|s)$", text_color=WHITE, fontsize=7)

# PPO Critic
draw_box(ax, (12.4, 2.0), 1.2, 1.0, "PPO Critic", ORANGE_DARK,
         sublabel=r"$V(s)$", text_color=WHITE, fontsize=7)

# Arrows from fusion to PPO
arrow(ax, (11.9, 4.0), (12.4, 4.1), color=ORANGE_DARK)
arrow(ax, (11.9, 3.0), (12.4, 2.5), color=ORANGE_DARK)

# Output arrows
arrow(ax, (13.6, 4.1), (13.9, 4.1), color=DARK, style="->")
ax.text(13.55, 4.45, r"$a_t$", ha="center", fontsize=7, color=DARK)
arrow(ax, (13.6, 2.5), (13.9, 2.5), color=DARK, style="->")


# ═══════════════════════════════════════════════════════════════════════════
# LOSS EQUATION (bottom)
# ═══════════════════════════════════════════════════════════════════════════

ax.text(7.0, 0.3,
        r"$\mathcal{L} = \mathcal{L}_\mathrm{RL} "
        r"+ \lambda_\mathrm{phys}\,\mathcal{L}_\mathrm{phys} "
        r"+ \lambda_\mathrm{reg}\,\mathcal{L}_\mathrm{reg}$",
        ha="center", va="center", fontsize=9, color=DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=GRAY_LIGHT,
                  edgecolor=GRAY, linewidth=1),
        zorder=5)


# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════

plt.tight_layout(pad=0.2)

if "--pdf" in sys.argv:
    plt.savefig("fig1_system_overview.pdf", bbox_inches="tight", dpi=300)
    print("Saved fig1_system_overview.pdf")
else:
    plt.savefig("fig1_system_overview.png", bbox_inches="tight", dpi=300)
    print("Saved fig1_system_overview.png")

plt.close()
