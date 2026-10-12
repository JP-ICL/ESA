"""Figure for the ESA report: the support function as the Legendre-dual picture of N^1_+ dA.

Top row: the sets A of fig_normal_bundle.py with their convex hull (dashed) and the
supporting line in the direction n = (0, 1). For the non-convex set the line touches A
at two points (the humps), while the tangent line at the dent point, which has the same
local normal, cuts through A.
Bottom row: the lift N^1_+ dA projected to (theta, h), h(t) = <n(t), x(t)>, i.e. the
multivalued support function over S^1. For the convex set it is a smooth graph; for the
non-convex set it is a swallowtail with cusps at the zero-curvature points, and the true
support function h_A = h_{conv A} is its upper envelope.
Output: fig_support_function.pdf next to this script.   Requires numpy, scipy, matplotlib.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
C_NB, C_DA, C_HL, C_HULL = "tab:red", "0.25", "tab:blue", "tab:green"


def polar_curve(r, dr, t):
    x = np.stack([r(t) * np.cos(t), r(t) * np.sin(t)], axis=-1)
    dx = np.stack([dr(t) * np.cos(t) - r(t) * np.sin(t),
                   dr(t) * np.sin(t) + r(t) * np.cos(t)], axis=-1)
    n = np.stack([dx[:, 1], -dx[:, 0]], axis=-1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    theta = np.unwrap(np.arctan2(n[:, 1], n[:, 0]))
    return x, n, theta


shapes = {
    "convex": (lambda t: 1 + 0.15 * np.cos(2 * t), lambda t: -0.30 * np.sin(2 * t)),
    "non-convex": (lambda t: 1 + 0.45 * np.cos(2 * t), lambda t: -0.90 * np.sin(2 * t)),
}
t_fine = np.linspace(0, 2 * np.pi, 2000)
th_star = np.pi / 2
n_star = np.array([np.cos(th_star), np.sin(th_star)])

fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2), gridspec_kw={"height_ratios": [2.2, 1]})
for col, (name, (r, dr)) in enumerate(shapes.items()):
    x, n, theta = polar_curve(r, dr, t_fine)
    h = np.einsum("ij,ij->i", n, x)                      # h(t) = <n(t), x(t)>
    th_grid = np.linspace(0, 2 * np.pi, 721)
    n_grid = np.stack([np.cos(th_grid), np.sin(th_grid)], axis=-1)
    h_conv = (n_grid @ x.T).max(axis=1)                  # support function of A (= of conv A)

    # boundary points sharing the normal n_star
    g = theta - th_star
    idx = np.where(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]
    t_rep = t_fine[idx] - g[idx] * (t_fine[idx + 1] - t_fine[idx]) / (g[idx + 1] - g[idx])
    xr, nr, _ = polar_curve(r, dr, t_rep)
    h_rep = xr @ n_star
    h_star = x @ n_star
    h_star = h_star.max()

    # --- top: the set, its hull and the supporting line ---------------------------
    ax = axes[0, col]
    ax.fill(x[:, 0], x[:, 1], color="0.88", zorder=0)
    hull = ConvexHull(x)
    hv = np.append(hull.vertices, hull.vertices[0])
    ax.plot(x[hv, 0], x[hv, 1], color=C_HULL, lw=1.3, ls="--", zorder=1)
    ax.plot(x[:, 0], x[:, 1], color=C_DA, lw=2.0, zorder=2)
    ax.axhline(h_star, color=C_HL, lw=1.4, zorder=3)
    ax.text(1.98, h_star + 0.05, r"$\langle n,y\rangle=h_A(n)$", color=C_HL, ha="right", fontsize=9)
    if len(t_rep) > 1:
        dent = np.argmin(h_rep)
        ax.axhline(h_rep[dent], color=C_HL, lw=1.0, ls=":", zorder=3)
        ax.text(1.98, h_rep[dent] - 0.06, "tangent at the dent,\nnot supporting", color=C_HL,
                ha="right", va="top", fontsize=8)
    ax.quiver(xr[:, 0], xr[:, 1], nr[:, 0], nr[:, 1], angles="xy", scale_units="xy",
              scale=1 / 0.3, width=0.007, color=C_HL, zorder=4)
    ax.plot(xr[:, 0], xr[:, 1], "o", ms=6, mfc="white", mec=C_HL, mew=1.8, zorder=5)
    ax.text(0, -0.15, r"$A$", ha="center", va="center", fontsize=14)
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 2.0); ax.set_ylim(-1.55, 1.75)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"({'ab'[col]}) {name} $A$: supporting line for $n=(0,1)$", fontsize=11)
    ax.legend(handles=[Line2D([], [], color=C_DA, lw=2.0, label=r"$\partial A$"),
                       Line2D([], [], color=C_HULL, lw=1.3, ls="--", label=r"$\partial\,\mathrm{conv}A$")],
              loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=9)

    # --- bottom: the support function over S^1 ------------------------------------
    ax = axes[1, col]
    ax.plot(th_grid / (2 * np.pi), h_conv, color=C_HULL, lw=3.2, alpha=0.45, zorder=1,
            label=r"$h_A(n)=\max_{y\in A}\langle n,y\rangle$")
    ax.plot(theta / (2 * np.pi), h, color=C_NB, lw=1.6, zorder=2,
            label=r"$N^1_+\partial A$ in $(\theta,\langle n,x\rangle)$")
    ax.axvline(th_star / (2 * np.pi), color=C_HL, lw=1.0, ls=":", zorder=1)
    ax.plot(np.full_like(h_rep, th_star / (2 * np.pi)), h_rep, "o", ms=6, mfc="white", mec=C_HL,
            mew=1.8, zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0.0, 1.65)
    ax.set_xlabel(r"normal angle $\theta/2\pi$")
    ax.set_ylabel(r"$\langle n(\theta),x\rangle$")
    ax.set_title(f"({'cd'[col]}) support function over $\\mathbb{{S}}^1$", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_support_function.pdf"))
fig.savefig(os.path.join(os.environ.get("PREVIEW_DIR", HERE), "fig_support_function.png"), dpi=110)
print("wrote fig_support_function.pdf")
