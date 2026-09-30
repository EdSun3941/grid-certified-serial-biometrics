"""Exact-dual certificate: solve the convexified root QP, recover multipliers from its KKT system (NNLS),
and evaluate the Lagrangian dual Z_D(mu) exactly (valid bound by weak duality, independent of QP accuracy)."""
import sys, time, numpy as np
sys.path.insert(0, "..")
from scipy.optimize import minimize, nnls
from experiment_core import fitting_data, make_blocks, B_DEFAULT
from envelope_solvers import fit_lr, LRNode, _design
from data import MATCHERS, data_file

def convex_qp_mu(Phi, y, N, acap, u, tol=1e-9):
    P = Phi * acap[None, :]; m = P.shape[1]
    s = np.maximum(P.max(axis=0), 1e-300); Q = P / s[None, :]
    H = Q.T @ Q; f = Q.T @ y
    fun = lambda v: float(v @ H @ v - 2 * f @ v + y @ y); jac = lambda v: 2 * (H @ v - f)
    capped = np.all(np.isfinite(u))
    cons = [{"type": "ineq", "fun": lambda v: Q @ v - y, "jac": lambda v: Q},
            {"type": "ineq", "fun": lambda v: N - (v / s).sum(), "jac": lambda v: -1.0 / s}]
    if capped: cons.append({"type": "ineq", "fun": lambda v: u - Q @ v, "jac": lambda v: -Q})
    best = None
    for v0 in [np.full(m, 1.0 / m), np.zeros(m)]:
        r = minimize(fun, v0, jac=jac, bounds=[(0, N * sb) for sb in s], constraints=cons, method="SLSQP", options={"ftol": 1e-15, "maxiter": 5000})
        if best is None or r.fun < best.fun: best = r
    v = best.x; yh = Q @ v; res = yh - y
    # KKT: 2Q^T res = Q_A^T lam_A - Q_C^T lam_C - nu/s + kappa_Z   (all >= 0)
    A = np.where(res <= tol * np.maximum(1, y))[0]
    Z = np.where(v <= 1e-12 * np.maximum(1, s))[0]
    C = np.where(capped & (u - yh <= tol))[0] if capped else np.array([], int)
    cols = [Q[A].T, -Q[C].T if len(C) else np.zeros((m, 0)), -(1.0 / s)[:, None]]
    E = np.zeros((m, len(Z))); E[Z, np.arange(len(Z))] = 1.0; cols.append(E)
    M = np.hstack(cols); rhs = 2 * Q.T @ res
    z, rn = nnls(M, rhs, maxiter=5000)
    lam = np.zeros(len(y)); lam[A] = z[:len(A)]
    lamC = np.zeros(len(y)); lamC[C] = z[len(A):len(A) + len(C)]
    mu = lam - lamC - 2 * res
    return best.fun, mu, rn

for ds, mm in [("fing_x_face", "face_C"), ("fing_x_face", "ri_V"), ("fing_x_fing", "li_V"), ("face_x_face", "face_G")]:
    D = dict(np.load(data_file(ds))); tr, te = make_blocks(D, [mm], 0); del D; G = tr["_G"]
    S = tr[mm][0]; x, y, _, _ = fitting_data(S[G], S[~G])
    _, U, info = fit_lr(x, y, B_DEFAULT, 2, "P1", mode="bb", time_limit=300)
    Phi, yv = _design(x, y, B_DEFAULT, False)
    for N in [2, 5]:
        for caps in [False, True]:
            node = LRNode(Phi, yv, "P1", N, obbt=caps); acap, u = node.caps(U)
            t = time.time(); qpv, mu, rn = convex_qp_mu(Phi, yv, N, acap, u); tq = time.time() - t
            z1, _, _ = node.run([(0, len(B_DEFAULT) - 1)] * N, mu, U, 1)
            print(f"{ds} {mm} N={N} caps={caps}: U={U:.6g} QP={qpv:.6g}  exact Z_D(mu_KKT)={z1:.6g} gap {100*(U-z1)/U:.4f}%  KKT resid {rn:.2e}  t {tq:.2f}s", flush=True)
