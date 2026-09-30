"""Independent reference for the N = 2 envelope problems on the exponent grid (round-2 revision).
Shares no optimization code with envelope_solvers.py.

P1: for every exponent pair b1 < b2 the problem  min ||A c - y||^2  s.t.  A c >= y, 0 <= c <= cbar  is a convex QP in
two variables.  It is solved exactly: the feasible polygon is built by clipping the box with every half-plane
(Sutherland-Hodgman), the unconstrained least-squares point is taken if it is feasible, and otherwise the minimum over
the polygon edges is computed in closed form (a convex function attains its constrained minimum on the boundary
when the unconstrained minimizer is infeasible).  Single exponents are covered by the edges c2 = 0.
P2: for every pair, the linear program  min sigma  s.t.  y <= A c <= y + sigma, 0 <= c <= cbar  is solved by HiGHS on
columns scaled to a maximum of one.
Columns are scaled to a maximum of one in both cases.  The returned value is evaluated at a point that dominates
every fitting point exactly (after a minimal upward rescaling if rounding requires it).
Usage: python reference_exact.py <dataset> <seed>   ->  results/E2ref/ref_<dataset>_s<seed>.csv"""
import itertools, os, sys, time
import numpy as np, pandas as pd
from scipy.optimize import linprog

CBAR = 1.0

def _clip(poly, a, b, yi):
    """Keep the part of polygon `poly` (list of 2-D points) with a*u1 + b*u2 >= yi."""
    out = []; n = len(poly)
    for k in range(n):
        P = poly[k]; Q = poly[(k + 1) % n]
        gP = a * P[0] + b * P[1] - yi; gQ = a * Q[0] + b * Q[1] - yi
        if gP >= 0: out.append(P)
        if (gP >= 0) != (gQ >= 0):
            t = gP / (gP - gQ); out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
    return out

def _evaluate(A, y, u):
    """Exact value at u, after the minimal upward rescaling that restores dominance (rounding only)."""
    g = A @ u
    if np.any(g < y):
        u = u * np.max(y / np.maximum(g, 1e-300)); g = A @ u
    return float(np.sum((g - y) ** 2)), u

def qp2_exact(A, y, ub):
    """min ||A u - y||^2 s.t. A u >= y, 0 <= u <= ub (A: n x 2, scaled columns)."""
    poly = [(0.0, 0.0), (ub[0], 0.0), (ub[0], ub[1]), (0.0, ub[1])]
    order = np.argsort(-y)                        # clip with the most restrictive constraints first
    for i in order:
        poly = _clip(poly, A[i, 0], A[i, 1], y[i])
        if not poly: return np.inf, None
    V = np.array(poly)
    u_ls, *_ = np.linalg.lstsq(A, y, rcond=None)
    if np.all(u_ls >= 0) and np.all(u_ls <= ub) and np.all(A @ u_ls >= y):
        return _evaluate(A, y, u_ls)
    best = (np.inf, None)
    for k in range(len(V)):
        P, Q = V[k], V[(k + 1) % len(V)]; d = Q - P; Ad = A @ d; r0 = A @ P - y
        den = float(Ad @ Ad)
        t = 0.0 if den <= 0 else min(1.0, max(0.0, -float(r0 @ Ad) / den))
        cand = P + t * d
        for u in (cand, P):
            v, uu = _evaluate(A, y, np.clip(u, 0, None))
            if v < best[0]: best = (v, uu)
    return best

def lp2(A, y, ub):
    n = len(y); c = np.r_[0.0, 0.0, 1.0]
    A_ub = np.vstack([np.hstack([-A, np.zeros((n, 1))]), np.hstack([A, -np.ones((n, 1))])])
    res = linprog(c, A_ub=A_ub, b_ub=np.r_[-y, y], bounds=[(0, ub[0]), (0, ub[1]), (0, None)], method="highs")
    if res.status != 0: return np.inf, None
    u = res.x[:2]; g = A @ u
    if np.any(g < y): u = u * np.max(y / np.maximum(g, 1e-300)); g = A @ u
    return float(np.max(g - y)), u

def reference(x, y, B, obj):
    Phi = x[:, None] ** np.asarray(B)[None, :]; s = Phi.max(axis=0); Q = Phi / s[None, :]
    best = (np.inf, None)
    for i, j in itertools.combinations(range(len(B)), 2):
        A = Q[:, [i, j]]; ub = CBAR * s[[i, j]]
        if np.any(A @ ub < y): continue                  # infeasible pair even at the caps
        v, u = (qp2_exact if obj == "P1" else lp2)(A, y, ub)
        if v < best[0]: best = (v, (B[i], B[j], u[0] / s[i], u[1] / s[j]))
    return best

if __name__ == "__main__":
    from experiment_core import fitting_data, make_blocks, B_DEFAULT
    from data import MATCHERS, data_file
    ds, seed = sys.argv[1], int(sys.argv[2])
    D = dict(np.load(data_file(ds))); M = MATCHERS[ds]; tr, _ = make_blocks(D, M, seed); del D; G = tr["_G"]
    rows = []; os.makedirs("../results/E2ref", exist_ok=True)
    for m in M:
        S = tr[m][0]; x, y, _, _ = fitting_data(S[G], S[~G])
        for obj in ["P1", "P2"]:
            t = time.time(); v, sol = reference(x, y, B_DEFAULT, obj); el = time.time() - t
            rows.append(dict(dataset=ds, seed=seed, matcher=m, obj=obj, N=2, method="Reference (exact pairs)", value=v,
                             b1=sol[0], b2=sol[1], c1=sol[2], c2=sol[3], time=el, n_fit=len(x)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"../results/E2ref/ref_{ds}_s{seed}.csv", index=False)
