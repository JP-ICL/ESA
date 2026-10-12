"""Numerics for the ESA report: boundary map of x_{i+1} = L x_i + xi_i,
L = diag(l1, l2), 0 < l1 < l2 < 1, spherical bounded noise of radius eps.

Coordinates on R^2 x S^1: z = (x1, x2, theta), n = (cos theta, sin theta),
    beta(x, n) = (L x + eps * Lperp(n), Lperp(n)),   Lperp(n) = L^{-T} n / |L^{-T} n|.

Checks performed:
  1. fixed points p_pm, q_pm; spectra and eigenvectors of D beta there;
  2. W^u(q_pm), computed by fundamental-domain iteration, agrees with the series
     x(n) = eps * sum_k L^k Lperp^{-k}(n)   (Lamb-Rasmussen-Tey, Thm 4.3);
  3. the conformal contact property beta^* alpha = alpha / |L^{-T} n|, alpha = n . dx;
  4. radius of curvature of dA at p_+ equals R = eps l1/(l1 - l2^2) when l1 > l2^2,
     and diverges (log at resonance, power law beyond) when l1 <= l2^2;
  5. at resonance l1 = l2^2, D beta(p_+) has a 2x2 Jordan block.
Figures: fig_boundary_map.pdf, fig_curvature.pdf.   Requires jax, numpy, matplotlib.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
OUT = "/home/claude/esa/"


def make_beta(l1, l2, eps):
    L = jnp.diag(jnp.array([l1, l2]))
    LinvT = jnp.linalg.inv(L).T

    def beta(z):
        x, th = z[:2], z[2]
        v = LinvT @ jnp.array([jnp.cos(th), jnp.sin(th)])
        th_new = jnp.arctan2(v[1], v[0])
        return jnp.concatenate([L @ x + eps * jnp.array([jnp.cos(th_new), jnp.sin(th_new)]), th_new[None]])

    return beta


def fixed_points(l1, l2, eps):
    return {
        "p+": jnp.array([eps / (1 - l1), 0.0, 0.0]),
        "p-": jnp.array([-eps / (1 - l1), 0.0, jnp.pi]),
        "q+": jnp.array([0.0, eps / (1 - l2), jnp.pi / 2]),
        "q-": jnp.array([0.0, -eps / (1 - l2), -jnp.pi / 2]),
    }


def series_x(theta, l1, l2, eps, K=150):   # K kept moderate to avoid underflow of l1^(2k)
    """x(n(theta)) = eps * sum_{k<K} L^k P(L^k n): the boundary point with outward normal n."""
    k = jnp.arange(K)
    a, b = l1**k, l2**k                                         # diagonal of L^k
    c, s = jnp.cos(theta), jnp.sin(theta)
    nrm = jnp.sqrt((a * c) ** 2 + (b * s) ** 2)
    return eps * jnp.array([jnp.sum(a * a * c / nrm), jnp.sum(b * b * s / nrm)])


def unstable_manifold(beta, q, n_pts=400, delta=1e-7, n_iter=80):
    w, V = np.linalg.eig(np.array(jax.jacfwd(beta)(q)))
    k = np.argmax(np.abs(w))
    kappa, vu = w[k].real, V[:, k].real
    f = jax.jit(jax.vmap(beta))
    branches = []
    for sgn in (+1, -1):
        t = np.geomspace(delta / kappa, delta, n_pts)          # fundamental domain
        pts = np.array(q)[None, :] + sgn * t[:, None] * vu[None, :]
        arcs = [pts]
        for _ in range(n_iter):
            pts = np.array(f(jnp.array(pts)))
            arcs.append(pts)
        branches.append(np.concatenate(arcs))
    return branches


def report(l1, l2, eps):
    beta = make_beta(l1, l2, eps)
    J = jax.jacfwd(beta)
    mu, rho = l1 / l2, l2**2 / l1
    print(f"\n=== l1={l1}, l2={l2}, eps={eps}: mu=l1/l2={mu:.4f}, l2^2/l1={rho:.4f}")
    for name, z in fixed_points(l1, l2, eps).items():
        w, V = np.linalg.eig(np.array(J(z)))
        target = mu if name[0] == "p" else 1 / mu
        k = np.argmin(np.abs(w - target))
        vec = V[:, k].real / V[2, k].real if abs(V[2, k]) > 1e-8 else V[:, k].real
        print(f"  {name}: |beta(z)-z|={float(jnp.max(jnp.abs(beta(z)-z))):.1e}, "
              f"spec={np.round(np.sort(w.real), 5)}, eigvec(theta-dir)={np.round(vec, 5)}")
    if l1 != l2**2:
        print(f"  R  = eps l1/(l1-l2^2) = {eps*l1/(l1-l2**2):.5f};   R~ = eps l2/(l2-l1^2) = {eps*l2/(l2-l1**2):.5f}")
    return beta


# ---------------------------------------------------------------- main example
L1, L2, EPS = 0.3, 0.5, 1.0          # normally hyperbolic regime: l1 > l2^2
beta = report(L1, L2, EPS)
report(0.4, 0.7, 1.0)               # l1 < l2^2

fps = fixed_points(L1, L2, EPS)
Wu = {name: unstable_manifold(beta, fps[name]) for name in ("q+", "q-")}
xs = jax.vmap(lambda t: series_x(t, L1, L2, EPS))
err = max(float(np.max(np.abs(np.array(xs(jnp.array(b[:, 2]))) - b[:, :2]))) for brs in Wu.values() for b in brs)
print(f"\nmax |W^u(q_pm) - series x(n)| = {err:.2e}")

# contact property at random points
key = jax.random.PRNGKey(1)
Z = jax.random.normal(key, (200, 3))
Lmat = jnp.diag(jnp.array([L1, L2]))
def contact_defect(z):
    th = z[2]
    n = jnp.array([jnp.cos(th), jnp.sin(th)])
    Db = jax.jacfwd(beta)(z)
    th1 = beta(z)[2]
    n1 = jnp.array([jnp.cos(th1), jnp.sin(th1)])
    pullback = n1 @ Db[:2, :]                                 # (beta^* alpha) as a row vector in (dx1,dx2,dtheta)
    alpha = jnp.array([n[0], n[1], 0.0])
    return jnp.max(jnp.abs(pullback - alpha / jnp.linalg.norm(jnp.linalg.inv(Lmat).T @ n)))
print(f"max |beta^*alpha - alpha/|L^-T n|| over 200 random points = {float(jnp.max(jax.vmap(contact_defect)(Z))):.2e}")

# resonance
report(0.49, 0.7, 1.0)
Jres = np.array(jax.jacfwd(make_beta(0.49, 0.7, 1.0))(fixed_points(0.49, 0.7, 1.0)["p+"]))
print("  resonant D beta(p+) =", np.round(Jres, 4).tolist(),
      " rank(J - l2 I) =", np.linalg.matrix_rank(Jres - 0.7 * np.eye(3), tol=1e-9))

# radius of curvature r(theta) = |dx/dtheta| near p_+ in three regimes
regimes = [(0.3, 0.5, r"$\lambda_1>\lambda_2^2$ ($0.3,\,0.5$)"),
           (0.25, 0.5, r"$\lambda_1=\lambda_2^2$ ($0.25,\,0.5$)"),
           (0.2, 0.5, r"$\lambda_1<\lambda_2^2$ ($0.2,\,0.5$)")]
thetas = jnp.geomspace(1e-10, 1e-1, 60)
curv = {}
for l1, l2, lab in regimes:
    dx = jax.vmap(jax.jacfwd(lambda t: series_x(t, l1, l2, 1.0, K=150)))(thetas)
    curv[lab] = np.array(jnp.linalg.norm(dx, axis=1))
    if l1 > l2**2:
        print(f"  r(1e-10) = {curv[lab][0]:.6f}  vs  R = {l1/(l1-l2**2):.6f}")
    if l1 < l2**2:
        a = np.log(l2**2 / l1) / np.log(l2 / l1)
        slope = np.polyfit(np.log(np.array(thetas[:30])), np.log(curv[lab][:30]), 1)[0]
        print(f"  power-law slope of r(theta): fitted {slope:.4f}, predicted -alpha = {-a:.4f}")
    if l1 == l2**2:
        slope = np.polyfit(np.log(np.array(thetas[:30])), curv[lab][:30], 1)[0]
        print(f"  log-law: dr/dlog(theta) fitted {slope:.4f}, predicted -1/log(l2/l1) = {-1/np.log(l2/l1):.4f}")

# ---------------------------------------------------------------- figures
names = {"p+": r"$p_+$", "p-": r"$p_-$", "q+": r"$q_+$", "q-": r"$q_-$"}
cols = {"q+": "tab:red", "q-": "tab:blue"}
th = jnp.linspace(0, 2 * jnp.pi, 800)
fig = plt.figure(figsize=(9.5, 4.4))
ax = fig.add_subplot(1, 2, 1)
for K in (1, 2, 3, 5):
    tb = np.array(jax.vmap(lambda t: series_x(t, L1, L2, EPS, K=K))(th))
    ax.plot(tb[:, 0], tb[:, 1], c="0.72", lw=0.8)
bd = np.array(xs(th))
ax.plot(bd[:, 0], bd[:, 1], c="k", lw=3, alpha=0.2)
for name, brs in Wu.items():
    for b in brs:
        ax.plot(b[:, 0], b[:, 1], "-", lw=1.3, c=cols[name])
for name, z in fps.items():
    z = np.array(z)
    sink = name[0] == "p"
    ax.annotate("", xy=(z[0] + 0.5 * np.cos(z[2]), z[1] + 0.5 * np.sin(z[2])), xytext=(z[0], z[1]),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1))
    ax.plot(z[0], z[1], "o" if sink else "s", c="k", ms=6, mfc="k" if sink else "w", zorder=5)
    off = {"p+": (5, 6), "p-": (-20, 6), "q+": (8, -14), "q-": (8, 4)}[name]
    ax.annotate(names[name], (z[0], z[1]), textcoords="offset points", xytext=off)
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.6, 2.6); ax.set_aspect("equal")
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
ax.set_title(r"(a) $\partial A$ and $\partial\mathcal{L}_\varepsilon^k(0)$, $k=1,2,3,5$", fontsize=10)

ax = fig.add_subplot(1, 2, 2, projection="3d")
wrap = lambda t: np.mod(t + np.pi / 2, 2 * np.pi) - np.pi / 2       # theta in [-pi/2, 3pi/2)
for name, brs in Wu.items():
    for b in brs:
        tw = wrap(b[:, 2])
        if name == "q-":                                            # q- sits on the seam
            hi = np.median(tw) > np.pi / 2
            tw = np.where(hi, np.where(tw < 0, tw + 2 * np.pi, tw), np.where(tw > np.pi, tw - 2 * np.pi, tw))
        ax.plot(b[:, 0], b[:, 1], tw, "-", lw=1.3, c=cols[name])
for name, z in fps.items():
    z = np.array(z)
    for t in [wrap(z[2])] + ([3 * np.pi / 2] if name == "q-" else []):
        ax.scatter(z[0], z[1], t, c="k" if name[0] == "p" else "w", edgecolors="k", s=30, depthshade=False)
        ax.text(z[0], z[1], t + 0.3, names[name], fontsize=9)
ax.set_zticks([-np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2])
ax.set_zticklabels([r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"])
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_zlabel(r"$\theta$")
ax.view_init(elev=18, azim=-35)
ax.set_title(r"(b) $N^1_+\partial A$: $W^u(q_+)$ (red), $W^u(q_-)$ (blue)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT + "fig_boundary_map.pdf")

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for (lab, r), c in zip(curv.items(), ["tab:green", "tab:orange", "tab:purple"]):
    ax.loglog(np.array(thetas), r, c=c, lw=1.6, label=lab)
ax.set_xlabel(r"$\theta$ (normal angle from $e_1$)")
ax.set_ylabel(r"radius of curvature $|x'(\theta)|$")
ax.set_title(r"Curvature of $\partial A$ near $p_+$ ($\varepsilon=1$)", fontsize=10)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT + "fig_curvature.pdf")
print("figures saved")
