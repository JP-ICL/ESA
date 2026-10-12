"""Companion checks for the ESA report (sections 3.5 and 4.1):
  * contact spectral pairing  spec(D beta) = {rho} u {nu, rho/nu}  at fixed points of boundary maps
    of a non-diagonal linear map, the Task 2 map f = (lam^2 x1 + a x2^2, lam x2) and a Henon-like map;
  * complex-eigenvalue windows at p_+ (0 < a eps < 2 lam^2) and p_- (-2 lam^2 < a eps < 0);
  * the extra fixed points n1 = -lam (1-lam)^2 / (2 a eps) and their spectra.
Requires jax, numpy.
"""
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
eps = 0.5
def make(f):
    Df = jax.jacfwd(f)
    def beta(z):
        x, th = z[:2], z[2]
        v = jnp.linalg.inv(Df(x)).T @ jnp.array([jnp.cos(th), jnp.sin(th)])
        t = jnp.arctan2(v[1], v[0])
        return jnp.concatenate([f(x) + eps*jnp.array([jnp.cos(t), jnp.sin(t)]), t[None]])
    def rho(z):
        x, th = z[:2], z[2]
        return 1/jnp.linalg.norm(jnp.linalg.inv(Df(x)).T @ jnp.array([jnp.cos(th), jnp.sin(th)]))
    return beta, rho
def newton(beta, z):
    G = lambda z: beta(z) - z
    for _ in range(60):
        z = z - jnp.linalg.solve(jax.jacfwd(G)(z), G(z))
    return z
cases = {
 "nondiag L": (lambda x: jnp.array([[0.3,0.15],[0.05,0.6]]) @ x, jnp.array([0.7,0.,0.])),
 "Task2 f, lam=0.6, a=0.4": (lambda x: jnp.array([0.36*x[0]+0.4*x[1]**2, 0.6*x[1]]), jnp.array([0.8,0.,0.])),
 "Task2 f, q+": (lambda x: jnp.array([0.36*x[0]+0.4*x[1]**2, 0.6*x[1]]), jnp.array([0.3,1.2,1.4])),
 "Henon-like": (lambda x: jnp.array([0.2*x[1] - 0.3*x[0]**2 + 0.1*x[0], 0.5*x[0] + 0.2*x[1]]), jnp.array([0.4,0.3,0.5])),
}
for name,(f,z0) in cases.items():
    beta, rho = make(f)
    z = newton(beta, z0)
    J = np.array(jax.jacfwd(beta)(z)); r = float(rho(z))
    th = float(z[2]); n = np.array([np.cos(th), np.sin(th)])
    # contact plane xi = ker(n.dx): basis (t,0) with t perp n, and d_theta
    t = np.array([-n[1], n[0]])
    B = np.array([[t[0],0],[t[1],0],[0,1]])            # columns span xi
    # project J restricted to xi back onto xi (J preserves xi at fixed point)
    alpha = np.array([n[0], n[1], 0.])
    print(f"{name}: z*={np.round(np.array(z),4)}, |beta-z|={np.max(np.abs(np.array(beta(z))-np.array(z))):.1e}")
    print("   alpha.J.xi (should be 0):", np.round(alpha @ J @ B, 12), "  alpha.J = rho*alpha?", np.allclose(alpha @ J, r*alpha))
    Jxi = np.linalg.lstsq(B, J @ B, rcond=None)[0]
    ev = np.linalg.eigvals(Jxi)
    print(f"   rho={r:.6f}, spec(J)={np.round(np.sort_complex(np.linalg.eigvals(J)),6)}, spec(J|xi)={np.round(ev,6)}, product={np.round(np.prod(ev),6)}")
print("---- complex window check, lam=0.6, eps=0.5 (2 lam^2 = 0.72 -> a in (0,1.44) for p+)")
lam=0.6
for a in [-1.6,-1.0,-0.2,0.2,1.0,1.4,1.5]:
    f = lambda x, a=a: jnp.array([lam**2*x[0]+a*x[1]**2, lam*x[1]])
    beta, rho = make(f)
    for nm, z in [("p+", jnp.array([eps/(1-lam**2),0.,0.])), ("p-", jnp.array([-eps/(1-lam**2),0.,jnp.pi]))]:
        ev = np.linalg.eigvals(np.array(jax.jacfwd(beta)(z)))
        print(f" a={a:5}: {nm} |beta-z|={np.max(np.abs(np.array(beta(z)-z))):.0e} complex={np.any(np.abs(ev.imag)>1e-12)} ev={np.round(ev,4)}")

print("---- extra fixed points, lam=0.6, eps=0.5, a=0.2")
lam, a = 0.6, 0.2
f = lambda x: jnp.array([lam**2*x[0]+a*x[1]**2, lam*x[1]])
beta, rho = make(f)
n1 = -lam*(1-lam)**2/(2*a*eps)
for sgn in (+1, -1):
    n2 = sgn*np.sqrt(1-n1**2); x2 = eps*n2/(1-lam); x1 = (eps*n1 + a*x2**2)/(1-lam**2)
    z = jnp.array([x1, x2, np.arctan2(n2, n1)])
    ev = np.linalg.eigvals(np.array(jax.jacfwd(beta)(z)))
    print(f"  n2 sign {sgn:+d}: z={np.round(np.array(z),4)}, |beta-z|={np.max(np.abs(np.array(beta(z)-z))):.1e}, rho={float(rho(z)):.4f}, spec={np.round(ev,4)}")
