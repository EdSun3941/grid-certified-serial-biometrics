"""Check reviewer R2-M1: is the Lagrangian dual of P1 tight when solved accurately?
Solve the convexified root QP with SLSQP, take mu = -2(yhat - y) and evaluate Z_D(mu) EXACTLY
(valid bound by weak duality), then polish with subgradient warm-started at that mu."""
import sys, time, numpy as np
sys.path.insert(0, "..")
from scipy.optimize import minimize
from experiment_core import fitting_data, make_blocks, B_DEFAULT
from envelope_solvers import fit_lr, fit_monomial, LRNode, _design
from data import MATCHERS, data_file

def convex_qp(Phi, y, N, acap, u):
    P = Phi * acap[None, :]; m = P.shape[1]
    s = np.maximum(P.max(axis=0), 1e-300); Q = P / s[None, :]
    H = Q.T @ Q; f = Q.T @ y
    fun = lambda v: float(v @ H @ v - 2 * f @ v + y @ y); jac = lambda v: 2 * (H @ v - f)
    cons = [{"type": "ineq", "fun": lambda v: Q @ v - y, "jac": lambda v: Q},
            {"type": "ineq", "fun": lambda v: N - (v / s).sum(), "jac": lambda v: -1.0 / s}]
    if np.all(np.isfinite(u)): cons.append({"type": "ineq", "fun": lambda v: u - Q @ v, "jac": lambda v: -Q})
    best = None
    for v0 in [np.full(m, 1.0 / m), np.zeros(m)]:
        r = minimize(fun, v0, jac=jac, bounds=[(0, N * sb) for sb in s], constraints=cons, method="SLSQP",
                     options={"ftol": 1e-15, "maxiter": 5000})
        if best is None or r.fun < best.fun: best = r
    return best.fun, Q @ best.x

cases = [("fing_x_face", "face_C"), ("fing_x_face", "ri_V"), ("fing_x_fing", "li_V"), ("face_x_face", "face_G")]
for ds, mm in cases:
    D = dict(np.load(data_file(ds))); tr, te = make_blocks(D, [mm], 0); del D; G = tr["_G"]
    S = tr[mm][0]; x, y, _, _ = fitting_data(S[G], S[~G])
    t = time.time(); _, U, info = fit_lr(x, y, B_DEFAULT, 2, "P1", mode="bb", time_limit=300); tbb = time.time() - t
    Phi, yv = _design(x, y, B_DEFAULT, False)
    for N in [2, 3, 4, 5]:
        for caps in [False, True]:
            node = LRNode(Phi, yv, "P1", N, obbt=caps)
            acap, u = node.caps(U)
            t = time.time(); qpv, yh = convex_qp(Phi, yv, N, acap, u); tq = time.time() - t
            mu = -2 * (yh - yv)
            z1, _, _ = node.run([(0, len(B_DEFAULT) - 1)] * N, mu, U, 1)
            z2, _, _ = node.run([(0, len(B_DEFAULT) - 1)] * N, mu, U, 300)
            print(f"{ds} {mm} N={N} caps={caps}: U={U:.6g} QPprimal={qpv:.6g} ({100*(U-qpv)/U:+.3f}%)  "
                  f"Z_D(mu_QP)={z1:.6g} gap {100*(U-z1)/U:.3f}%  +300 subgrad {100*(U-z2)/U:.3f}%  tQP {tq:.2f}s tBB(N=2) {tbb:.2f}s", flush=True)
