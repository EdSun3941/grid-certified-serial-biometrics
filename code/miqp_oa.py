"""IJIS revision (reviewer 2): general-purpose mixed-integer reference for P1 that shares no optimization code with
LR-BB (envelope_solvers.py) or with the pair enumeration (reference_exact.py).

P1 on the grid B is the convex MIQP
    min sum_i (yhat_i - y_i)^2   s.t.  yhat = Phi c >= y,  0 <= c_b <= cap_b z_b,  sum_b z_b <= N,  z binary,
with at most N distinct exponents and a per-exponent coefficient cap (cbar = 1, tightened by the cutoff argument with the
trivial feasible value U0 of c = 1 at b = 0).  It is solved by outer approximation (Duran and Grossmann 1986): the
separable objective is replaced by variables t_i >= 0 with tangent cuts t_i >= 2 e^k (e_i) - (e^k)^2 of e_i^2 at
e = yhat - y, the resulting MILP is solved by HiGHS (scipy.optimize.milp), and cuts are added at the MILP solution and
at the exact convex-QP solution for the selected support, until the MILP dual bound (a valid lower bound, since the cuts
under-estimate the objective) is within the relative tolerance of the best feasible value.
Columns are scaled to a maximum of one.  The QP on a fixed support (at most N columns) is solved by SLSQP; it only
produces upper bounds, so its accuracy does not affect the lower bound.
Usage: python miqp_oa.py <dataset> <seed> [N]   ->  results/E2miqp/miqp_<dataset>_s<seed>_N<N>.csv"""
import os, sys, time
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds, minimize
from scipy.sparse import csr_matrix, vstack, hstack, identity, diags

CBAR = 1.0


def _true_value(Q, y, v):
    g = Q @ v
    if np.any(g < y):                                  # minimal upward rescaling for exact dominance (rounding only)
        v = v * np.max(y / np.maximum(g, 1e-300)); g = Q @ v
    return float(np.sum((g - y) ** 2)), v


def _qp_support(Q, y, S, M):
    """min ||Q_S v - y||^2 s.t. Q_S v >= y, 0 <= v <= M_S (small convex QP; upper bounds only)."""
    A = Q[:, S]
    v0 = np.minimum(M[S], np.full(len(S), np.max(y) / max(A.max(), 1e-300)))
    for scale in (1.0, 2.0, 5.0):                      # a few starts; the best feasible point is kept
        x0 = np.minimum(M[S], v0 * scale)
        r = minimize(lambda v: np.sum((A @ v - y) ** 2), x0, jac=lambda v: 2 * A.T @ (A @ v - y), method="SLSQP",
                     bounds=[(0, m) for m in M[S]], constraints=[{"type": "ineq", "fun": lambda v: A @ v - y, "jac": lambda v: A}],
                     options={"maxiter": 500, "ftol": 1e-14})
        v = np.zeros(Q.shape[1]); v[S] = np.clip(r.x, 0, M[S])
        yield _true_value(Q, y, v)


def miqp_p1_oa(x, y, B, N, tol=1e-4, time_limit=600.0, max_iter=400):
    t0 = time.time()
    x = np.asarray(x, float); y = np.asarray(y, float); n, m = len(x), len(B)
    Phi = x[:, None] ** np.asarray(B)[None, :]; s = Phi.max(0); Q = Phi / s
    U0 = float(np.sum((1.0 - y) ** 2)); u = y + np.sqrt(U0)
    cap = np.minimum(CBAR, np.min(u[:, None] / Phi, axis=0)); M = cap * s            # bound on the scaled coefficient
    # variables: [v (m) | z (m) | t (n)]
    nv = 2 * m + n
    c = np.r_[np.zeros(2 * m), np.ones(n)]
    Qs = csr_matrix(Q)
    A_dom = hstack([Qs, csr_matrix((n, m)), csr_matrix((n, n))])                      # Q v >= y
    A_link = hstack([identity(m), -diags(M), csr_matrix((m, n))])                    # v - M z <= 0
    A_card = hstack([csr_matrix((1, m)), csr_matrix(np.ones((1, m))), csr_matrix((1, n))])  # sum z <= N
    base_A = vstack([A_dom, A_link, A_card]).tocsr()
    base_lb = np.r_[y, np.full(m, -np.inf), -np.inf]; base_ub = np.r_[np.full(n, np.inf), np.zeros(m), N]
    cut_rows, cut_lb = [], []

    def add_cuts(e):
        e = np.maximum(e, 0.0)
        # t_i - 2 e_i Q_i v >= -2 e_i y_i - e_i^2
        Av = -2 * e[:, None] * Q
        row = hstack([csr_matrix(Av), csr_matrix((n, m)), identity(n)])
        cut_rows.append(row); cut_lb.append(-2 * e * y - e ** 2)

    for e0 in (0.01, 0.05, 0.2):                       # a few initial tangents (valid anywhere)
        add_cuts(np.full(n, e0))
    best_ub, best_v, lb, it, status = np.inf, None, -np.inf, 0, "iteration limit"
    integrality = np.r_[np.zeros(m), np.ones(m), np.zeros(n)]
    bounds = Bounds(np.zeros(nv), np.r_[M, np.ones(m), np.full(n, np.inf)])
    seen = set()
    while it < max_iter:
        it += 1
        A = vstack([base_A] + cut_rows).tocsr()
        lo = np.r_[base_lb, np.concatenate(cut_lb)]; hi = np.r_[base_ub, np.full(sum(r.shape[0] for r in cut_rows), np.inf)]
        rem = time_limit - (time.time() - t0)
        if rem <= 1: status = "time limit"; break
        res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integrality, bounds=bounds,
                   options={"time_limit": rem, "mip_rel_gap": 1e-7, "disp": False})
        if res.x is None: status = f"milp failed ({res.message})"; break
        lb = max(lb, float(getattr(res, "mip_dual_bound", res.fun)))
        v = res.x[:m]; z = res.x[m:2 * m] > 0.5
        val, vv = _true_value(Q, y, np.where(z, v, 0.0))
        if val < best_ub: best_ub, best_v = val, vv
        add_cuts(Q @ v - y)
        S = tuple(np.nonzero(z)[0].tolist())
        if S and S not in seen:
            seen.add(S)
            for qv, qvv in _qp_support(Q, y, list(S), M):
                if qv < best_ub: best_ub, best_v = qv, qvv
                add_cuts(Q @ qvv - y)
        if best_ub <= lb * (1 + tol) + 1e-15 or (best_ub - lb) <= tol * best_ub:
            status = "optimal"; break
    sup = np.nonzero(best_v > 1e-15)[0] if best_v is not None else []
    terms = [(float(best_v[k] / s[k]), float(B[k])) for k in sup]
    return dict(value=best_ub, lb=lb, gap=(best_ub - lb) / best_ub if best_ub > 0 else np.nan, iters=it, status=status,
                time=time.time() - t0, terms=terms, n_supports=len(seen))


def main():
    from experiment_core import fitting_data, make_blocks, B_DEFAULT
    from data import MATCHERS, data_file
    ds, seed = sys.argv[1], int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    D = dict(np.load(data_file(ds))); M = MATCHERS[ds]; tr, _ = make_blocks(D, M, seed); del D; G = tr["_G"]
    rows = []
    for mt in M:
        S = tr[mt][0]; xf, yf, _, _ = fitting_data(S[G], S[~G])
        r = miqp_p1_oa(xf, yf, B_DEFAULT, N)
        rows.append({"dataset": ds, "seed": seed, "matcher": mt, "obj": "P1", "N": N, "method": "MIQP (OA + HiGHS)",
                     "value": r["value"], "lb": r["lb"], "gap": r["gap"], "iters": r["iters"], "status": r["status"],
                     "time": r["time"], "terms": str(r["terms"]), "n_supports": r["n_supports"], "n_fit": len(xf)})
        print(f"[miqp {ds} s{seed} N{N}] {mt} value={r['value']:.6g} lb={r['lb']:.6g} gap={r['gap']:.2e} it={r['iters']} "
              f"t={r['time']:.1f}s {r['status']}", flush=True)
    os.makedirs("../results/E2miqp", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"../results/E2miqp/miqp_{ds}_s{seed}_N{N}.csv", index=False)


if __name__ == "__main__":
    main()
