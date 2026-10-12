"""Figure for the ESA report: outward unit normal bundle N^1_+ dA of a planar set A.

Top row: boundary dA with the outward unit normal n(x) at sampled boundary points,
for a (strictly) convex set and a non-convex set. Bottom row: the angle theta(t) of
n(x(t)) along the boundary parameter t. For the convex set theta is monotone (the
Gauss map dA -> S^1 is a bijection, so N^1_+ dA is a graph over S^1); for the
non-convex set it is not.
Output: fig_normal_bundle.pdf next to this script.   Requires numpy, matplotlib.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def polar_curve(r, dr, t):
    """Position, outward unit normal and normal angle for x(t) = r(t)(cos t, sin t)."""
    x = np.stack([r(t) * np.cos(t), r(t) * np.sin(t)], axis=-1)
    dx = np.stack([dr(t) * np.cos(t) - r(t) * np.sin(t),
                   dr(t) * np.sin(t) + r(t) * np.cos(t)], axis=-1)
    n = np.stack([dx[:, 1], -dx[:, 0]], axis=-1)       # rotate tangent by -90 deg
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    theta = np.unwrap(np.arctan2(n[:, 1], n[:, 0]))
    return x, n, theta


shapes = {
    "convex": (lambda t: 1 + 0.15 * np.cos(2 * t), lambda t: -0.30 * np.sin(2 * t)),
    "non-convex": (lambda t: 1 + 0.45 * np.cos(2 * t), lambda t: -0.90 * np.sin(2 * t)),
}

t_fine = np.linspace(0, 2 * np.pi, 800)
t_arrow = np.linspace(0, 2 * np.pi, 72, endpoint=False)
arrow_len = 0.24
C_NB = "tab:red"      # everything that makes up N^1_+ dA = {(x, n(x))}
C_DA = "0.25"         # the boundary dA alone (the front of N^1_+ dA)
C_HL = "tab:blue"     # highlighted boundary points sharing one normal direction
from matplotlib.lines import Line2D

fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2), gridspec_kw={"height_ratios": [2.2, 1]})
for col, (name, (r, dr)) in enumerate(shapes.items()):
    ax = axes[0, col]
    x, _, _ = polar_curve(r, dr, t_fine)
    ax.fill(x[:, 0], x[:, 1], color="0.88", zorder=0)
    ax.plot(x[:, 0], x[:, 1], color=C_DA, lw=2.0, zorder=1)
    xa, na, _ = polar_curve(r, dr, t_arrow)
    ax.quiver(xa[:, 0], xa[:, 1], na[:, 0], na[:, 1], angles="xy", scale_units="xy",
              scale=1 / arrow_len, width=0.004, color=C_NB, zorder=2)
    ax.plot(xa[:, 0], xa[:, 1], "o", ms=1.8, color=C_NB, zorder=3)
    ax.legend(handles=[Line2D([], [], color=C_DA, lw=2.0, label=r"$\partial A$"),
                       Line2D([], [], color=C_NB, lw=1.5, marker=">", label=r"$N^1_+\partial A=\{(x,n(x))\}$")],
              loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=9)
    ax.text(0, 0, r"$A$", ha="center", va="center", fontsize=14)
    ax.set_aspect("equal")
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.55, 1.55)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"({'ab'[col]}) {name} $A$: $\\partial A$ and $n(x)$", fontsize=11)

    # points of dA sharing the same normal n = (0, 1) (angle theta* = pi/2); three of them when A is non-convex
    th_star = np.pi / 2
    _, _, th_fine = polar_curve(r, dr, t_fine)
    g = th_fine - th_star
    idx = np.where(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]
    t_rep = t_fine[idx] - g[idx] * (t_fine[idx + 1] - t_fine[idx]) / (g[idx + 1] - g[idx])
    xr, nr, _ = polar_curve(r, dr, t_rep)
    if len(t_rep) > 1:
        ax.quiver(xr[:, 0], xr[:, 1], nr[:, 0], nr[:, 1], angles="xy", scale_units="xy",
                  scale=1 / (1.3 * arrow_len), width=0.008, color=C_HL, zorder=4)
        ax.plot(xr[:, 0], xr[:, 1], "o", ms=6, mfc="white", mec=C_HL, mew=1.8, zorder=5)

    ax = axes[1, col]
    _, _, theta = polar_curve(r, dr, t_fine)
    ax.plot(t_fine / (2 * np.pi), theta / (2 * np.pi), color=C_NB, lw=1.8, label=r"$N^1_+\partial A$ in $(t,\theta)$")
    if len(t_rep) > 1:
        ax.axhline(th_star / (2 * np.pi), color=C_HL, lw=1.0, ls=":", zorder=1)
        ax.plot(t_rep / (2 * np.pi), np.full_like(t_rep, th_star / (2 * np.pi)), "o", ms=6,
                mfc="white", mec=C_HL, mew=1.8, zorder=5, label=r"same normal $n=(0,1)$")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.plot(t_fine / (2 * np.pi), t_fine / (2 * np.pi), color="0.6", lw=0.8, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("boundary parameter $t/2\\pi$")
    ax.set_ylabel(r"normal angle $\theta/2\pi$")
    ax.set_title(f"({'cd'[col]}) $N^1_+\\partial A$ as a curve in $(t,\\theta)$", fontsize=11)
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_normal_bundle.pdf"))
fig.savefig(os.path.join(os.environ.get("PREVIEW_DIR", HERE), "fig_normal_bundle.png"), dpi=110)
print("wrote fig_normal_bundle.pdf")
