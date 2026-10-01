"""Posynomial upper-envelope fitting: exact alpha solvers, Lagrangian relaxation (LR),
LR with optimality-based bound tightening (OBBT) and LR-based branch-and-bound (LR-BB),
plus baseline fitters.  Envelope g(a) = sum_j alpha_j a^{beta_j}, alpha>=0, beta<=0.

Objectives (same notation as the thesis):
  P1: min sum_i (g(x_i) - y_i)^2           s.t. g(x_i) >= y_i
  P2: min S                                s.t. y_i <= g(x_i) <= y_i + S
"""
import heapq, itertools, time
import numpy as np
from scipy.optimize import minimize, linprog, differential_evolution, least_squares

AMAX = 1.0  # global cap on alpha (a single term with alpha>1 exceeds FRR=1 everywhere on (0,1])

# ----------------------------------------------------------------------------- utilities
def env_eval(terms, a):
    a = np.asarray(a, float)
    return sum(al * a ** be for al, be in terms) if terms else np.zeros_like(a)

def objective(terms, x, y, obj):
    g = env_eval(terms, x); r = g - y
    return float(np.sum(r ** 2)) if obj == "P1" else float(np.max(r))

def dominates(terms, x, y, tol=1e-12):
    return bool(np.all(env_eval(terms, x) >= y - tol))

# ----------------------------------------------------------------------------- exact alpha
def solve_alpha(Phi, y, obj, amax=AMAX):
    """Exact convex problem in alpha for fixed exponents (columns of Phi).
    P1: QP (SLSQP); P2: LP (HiGHS). Returns (value, alpha) or (inf, None) if infeasible."""
    k = Phi.shape[1]
    colmax = Phi.max(axis=0)
    # feasibility: best achievable domination with alpha=amax on all columns
    if np.any(Phi @ np.full(k, amax) < y - 1e-12):
        return np.inf, None
    sc = np.maximum(colmax, 1e-300); Q = Phi / sc[None, :]          # columns scaled to max 1 (round-2 fix)
    if obj == "P2":
        c = np.zeros(k + 1); c[-1] = 1.0
        A_ub = np.vstack([np.hstack([-Q, np.zeros((len(y), 1))]), np.hstack([Q, -np.ones((len(y), 1))])])
        b_ub = np.concatenate([-y, y])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, amax * v) for v in sc] + [(0, None)], method="highs")
        if res.status != 0:
            return np.inf, None
        al = np.clip(res.x[:k] / sc, 0, amax)
        g = Phi @ al
        if np.any(g < y):  # tiny LP tolerance repair
            al = al * np.max(y / np.maximum(g, 1e-300))
        return float(np.max(Phi @ al - y)), al
    # P1: solved in scaled variables v = alpha * colmax (round-2 fix: unscaled columns made SLSQP fail)
    ub = amax * sc
    s0 = np.max(y / np.maximum(Q.sum(axis=1), 1e-300))
    starts = [np.minimum(np.full(k, s0), ub), ub.copy()]
    H = Q.T @ Q; f = Q.T @ y
    fun = lambda v: float(v @ H @ v - 2 * f @ v + y @ y)
    jac = lambda v: 2 * (H @ v - f)
    cons = [{"type": "ineq", "fun": lambda v: Q @ v - y, "jac": lambda v: Q}]
    best = None
    for v0 in starts:
        if np.any(Q @ v0 < y): continue
        res = minimize(fun, v0, jac=jac, bounds=list(zip(np.zeros(k), ub)), constraints=cons, method="SLSQP",
                       options={"ftol": 1e-15, "maxiter": 500})
        vv = np.clip(res.x, 0, ub); g = Q @ vv
        if np.any(g < y): vv = np.minimum(vv * np.max(y / np.maximum(g, 1e-300)), ub)
        val = float(np.sum((Q @ vv - y) ** 2)) if np.all(Q @ vv >= y - 1e-12) else np.inf
        if best is None or val < best[0]: best = (val, vv)
    if best is None or not np.isfinite(best[0]):
        return np.inf, None
    al = np.clip(best[1] / sc, 0, amax)
    g = Phi @ al
    if np.any(g < y):  # repair tolerance violations by minimal upward scaling
        al = np.minimum(al * np.max(y / np.maximum(g, 1e-300)), amax)
        if np.any(Phi @ al < y - 1e-12):
            return np.inf, None
    return float(np.sum((Phi @ al - y) ** 2)), al

class AlphaCache:
    """Caches exact alpha solutions per exponent subset (grid indices)."""
    def __init__(self, Phi, y, obj, amax=AMAX):
        self.Phi, self.y, self.obj, self.amax, self.c = Phi, y, obj, amax, {}
        self.n_solves = 0
    def __call__(self, idx):
        key = tuple(sorted(set(int(i) for i in idx)))
        if key not in self.c:
            self.n_solves += 1
            self.c[key] = solve_alpha(self.Phi[:, list(key)], self.y, self.obj, self.amax)
        return key, self.c[key]

# ----------------------------------------------------------------------------- baselines
def fit_hyperbola(x, y):
    C = float(np.max(x * y)); return [(C, -1.0)]

def _design(x, y, B, rel):
    """Design matrix and target. rel=True: relative (per-point scaled by 1/y_i) errors."""
    Phi = x[:, None] ** np.asarray(B)[None, :]
    if rel:
        return Phi / y[:, None], np.ones_like(y)
    return Phi, y

def objective_any(terms, x, y, obj, rel=False):
    g = env_eval(terms, x); r = (g - y) / y if rel else (g - y)
    return float(np.sum(r ** 2)) if obj == "P1" else float(np.max(r))

def fit_monomial(x, y, B, obj, rel=False):
    """Exact N=1 on the beta grid (closed form in alpha)."""
    best = (np.inf, None)
    yy = np.ones_like(y) if rel else y
    for b in B:
        phi = x ** b / y if rel else x ** b
        amin = np.max(yy / phi)
        if obj == "P1":
            al = max(amin, float(phi @ yy / (phi @ phi)))
        else:
            al = amin
        if al > AMAX: continue
        v = objective_any([(al, b)], x, y, obj, rel)
        if v < best[0]: best = (v, [(al, float(b))])
    return best[1], best[0]

def fit_enum(x, y, B, N, obj, time_limit=None, rel=False):
    """Exact reference: enumerate all subsets of N distinct grid exponents; exact alpha each."""
    Phi, yv = _design(x, y, B, rel)
    cache = AlphaCache(Phi, yv, obj); best = (np.inf, None); t0 = time.time(); done = True
    for comb in itertools.combinations(range(len(B)), N):
        key, (v, al) = cache(comb)
        if v < best[0]: best = (v, [(float(a), float(B[i])) for a, i in zip(al, key) if a > 0])
        if time_limit and time.time() - t0 > time_limit: done = False; break
    return best[1], best[0], {"n_qp": cache.n_solves, "complete": done, "time": time.time() - t0}

def fit_nls_shift(x, y, N, obj, seed=0, n_starts=10, rel=False):
    """Unconstrained least-squares posynomial fit (log-alpha, beta), then scaled up to dominate."""
    rng = np.random.default_rng(seed); best = (np.inf, None)
    w = 1.0 / y if rel else np.ones_like(y)
    def resid(p):
        la, be = p[:N], p[N:]
        return w * (np.exp(la)[None, :] * x[:, None] ** be[None, :] @ np.ones(N) - y)
    for s in range(n_starts):
        p0 = np.concatenate([np.log(rng.uniform(1e-3, 0.1, N)), rng.uniform(-1.5, 0, N)])
        try:
            r = least_squares(resid, p0, bounds=(np.r_[[-60] * N, [-3] * N], np.r_[[0] * N, [0] * N]))
        except Exception:
            continue
        terms = [(float(np.exp(a)), float(b)) for a, b in zip(r.x[:N], r.x[N:])]
        g = env_eval(terms, x); k = float(np.max(y / np.maximum(g, 1e-300)))
        terms = [(min(al * k, AMAX), be) for al, be in terms]
        if not dominates(terms, x, y): continue
        v = objective_any(terms, x, y, obj, rel)
        if v < best[0]: best = (v, terms)
    return best[1], best[0]

def fit_de(x, y, N, obj, seed=0, maxiter=200, popsize=15, rel=False):
    """Differential evolution (GA-type) over (log10 alpha, beta) with dominance penalty; final
    upward scaling guarantees dominance."""
    def unpack(p):
        return [(10 ** p[2 * j], p[2 * j + 1]) for j in range(N)]
    def f(p):
        t = unpack(p); g = env_eval(t, x)
        if rel: g, yy = g / y, np.ones_like(y)
        else: yy = y
        viol = np.maximum(yy - g, 0)
        base = np.sum((g - yy) ** 2) if obj == "P1" else np.max(g - yy)
        return base + 1e3 * np.sum(viol ** 2) + 10 * np.sum(viol)
    bounds = [(-16, 0), (-3, 0)] * N
    r = differential_evolution(f, bounds, seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-10, polish=True)
    terms = unpack(r.x); g = env_eval(terms, x); k = max(1.0, float(np.max(y / np.maximum(g, 1e-300))))
    terms = [(min(al * k, AMAX), be) for al, be in terms]
    return terms, objective_any(terms, x, y, obj, rel), {"nfev": int(r.nfev)}

def fit_maxmono(x, y, B, obj, K=3, rel=False):
    """GP-compatible piecewise-linear envelope in log-log space: g(a)=max_k m_k(a).
    Segments = K equal-count groups in FAR; each monomial dominates its own segment
    (exact N=1 fit), the max over pieces dominates all points."""
    o = np.argsort(x); parts = np.array_split(o, K); pieces = []
    for p in parts:
        t, _ = fit_monomial(x[p], y[p], B, obj, rel)
        pieces.append(t[0])
    g = np.max([al * x ** be for al, be in pieces], axis=0)
    val = float(np.sum((g - y) ** 2)) if obj == "P1" else float(np.max(g - y))
    return pieces, val

def maxmono_eval(pieces, a):
    return np.max([al * np.asarray(a, float) ** be for al, be in pieces], axis=0)

# ----------------------------------------------------------------------------- LR machinery
class LRNode:
    """Lagrangian relaxation of the curve-definition constraints on a node whose term j has
    exponent restricted to grid-index interval [l_j, h_j].  Valid lower bound at every mu."""
    def __init__(self, Phi, y, obj, N, obbt=True):
        self.Phi, self.y, self.obj, self.N, self.obbt = Phi, y, obj, N, obbt
        self.nD, self.nB = Phi.shape

    def caps(self, U):
        if not self.obbt or not np.isfinite(U):
            return np.full(self.nB, AMAX), np.full(self.nD, np.inf)
        u = self.y + (np.sqrt(U) if self.obj == "P1" else U)          # yhat_i <= u_i for any better solution
        acap = np.minimum(AMAX, np.min(u[:, None] / self.Phi, axis=0))  # each term <= u_i
        return acap, u

    def run(self, iv, mu, U, K, lam=2.0, patience=15, candidates=None):
        """Subgradient ascent for K iterations. Returns (best_ZD, mu_best, lam)."""
        y, Phi, N = self.y, self.Phi, self.N
        acap, u = self.caps(U)
        best, mu_best, stall, pat = -np.inf, mu.copy(), 0, patience
        target = U if np.isfinite(U) else None
        for k in range(K):
            if self.obj == "P1":
                m = mu
                yh = np.clip(y - m / 2, y, u)
                h = np.sum((yh - y) ** 2 + m * yh)
                c = m @ Phi                               # term cost: -alpha * c_b
                val = -acap * c
            else:
                m1, m2 = mu[: self.nD], mu[self.nD:]
                Sbar = U if np.isfinite(U) else 1.0
                coefS = 1.0 - m1.sum(); S = Sbar if coefS < 0 else 0.0
                h = coefS * S + np.sum((m2 - m1) * y)
                c = (m1 - m2) @ Phi                        # term cost: +alpha * c_b
                val = acap * c
            tsum = np.zeros(self.nD); zd = h; chosen = []
            for (l, hh) in iv:
                seg = val[l:hh + 1]; j = int(np.argmin(seg)); b = l + j
                if seg[j] < 0:
                    zd += seg[j]; tsum += acap[b] * Phi[:, b]; chosen.append(b)
            if candidates is not None:
                # LR-guided column selection for the primal heuristic: most negative reduced costs
                lo = min(l for l, _ in iv); hi = max(hh for _, hh in iv)
                order = lo + np.argsort(val[lo:hi + 1])[: max(N, 1)]
                candidates.add(tuple(sorted(set(order.tolist()))))
                if chosen: candidates.add(tuple(sorted(set(chosen))))
            if zd > best + 1e-15:
                best, mu_best, stall = zd, mu.copy(), 0
            else:
                stall += 1
                if stall >= pat:
                    lam /= 2; stall = 0; pat += 2
            if self.obj == "P1":
                g = yh - tsum
            else:
                g = np.concatenate([tsum - y - S, y - tsum])
            gg = float(g @ g)
            if gg < 1e-30 or lam < 1e-6: break
            tgt = target if target is not None else zd + abs(zd) * 0.1 + 1e-6
            step = lam * max(tgt - zd, 1e-12) / gg
            mu = mu + step * g
            if self.obj == "P2": mu = np.maximum(mu, 0)
        return best, mu_best, lam

def fit_lr(x, y, B, N, obj, mode="bb", time_limit=60.0, eps=1e-4, K_root=300, K_node=40,
           enum_leaf=16, obbt=True, init=None, log=None, rel=False):
    """mode='root': LR at the root only (thesis-style: LR bound + primal heuristic);
       mode='bb'  : LR-based branch-and-bound over exponent intervals (exact up to eps)."""
    t0 = time.time()
    Phi, yv = _design(x, y, B, rel); nB = len(B)
    cache = AlphaCache(Phi, yv, obj)
    node = LRNode(Phi, yv, obj, N, obbt=obbt)
    # incumbent: exact monomial (N=1) + LR-guided subsets
    U, best_key, best_al = np.inf, None, None
    def try_key(idx):
        nonlocal U, best_key, best_al
        key, (v, al) = cache(idx)
        if v < U: U, best_key, best_al = v, key, al
    if init is not None:
        try_key(init)
    mono_terms, mono_v = fit_monomial(x, y, B, obj, rel)
    if mono_terms: try_key([int(np.argmin(np.abs(B - mono_terms[0][1])))])
    mu0 = np.zeros(len(yv) if obj == "P1" else 2 * len(yv))
    root_iv = [(0, nB - 1)] * N
    cand = set()
    zd_root, mu_root, _ = node.run(root_iv, mu0, U, K_root, candidates=cand)
    for c in list(cand)[:200]: try_key(c)
    # re-run root with the improved incumbent (tighter OBBT caps & Polyak target)
    zd2, mu_root, _ = node.run(root_iv, mu_root, U, K_root, candidates=cand)
    for c in list(cand)[:400]: try_key(c)
    root_lb = max(zd_root, zd2)
    hist = {"root_lb": root_lb, "root_ub": U}
    if mode == "root":
        lb = root_lb
        out_terms = [(float(a), float(B[i])) for a, i in zip(best_al, best_key) if a > 0]
        return out_terms, U, {"lb": lb, "gap": (U - lb) / U if U > 0 else 0.0, "time": time.time() - t0,
                              "nodes": 1, "n_qp": cache.n_solves, "root_lb": root_lb, "root_ub": U}
    # branch-and-bound
    cnt = itertools.count()
    heap = [(root_lb, next(cnt), tuple(root_iv), mu_root)]
    nodes = 1; closed_lb = np.inf
    while heap:
        if time.time() - t0 > time_limit: break
        lb, _, iv, mu = heapq.heappop(heap)
        if lb >= U * (1 - eps) - 1e-15:
            closed_lb = min(closed_lb, lb); heap.clear(); break
        sizes = [h - l + 1 for l, h in iv]
        if int(np.prod(sizes, dtype=float)) <= enum_leaf:
            for comb in itertools.product(*[range(l, h + 1) for l, h in iv]):
                if all(comb[j] <= comb[j + 1] for j in range(N - 1)): try_key(comb)
            nodes += 1; continue
        cand = set()
        zd, mu2, _ = node.run(list(iv), mu, U, K_node, candidates=cand)
        for c in list(cand)[:20]: try_key(c)
        nodes += 1
        nlb = max(lb, zd)
        if nlb >= U * (1 - eps) - 1e-15: continue
        j = int(np.argmax(sizes)); l, h = iv[j]; m = (l + h) // 2
        left = list(iv); left[j] = (l, m)
        for k in range(j): left[k] = (left[k][0], min(left[k][1], m))
        right = list(iv); right[j] = (m + 1, h)
        for k in range(j + 1, N): right[k] = (max(right[k][0], m + 1), right[k][1])
        for ch in (left, right):
            if all(a <= b for a, b in ch):
                heapq.heappush(heap, (nlb, next(cnt), tuple(ch), mu2))
    # an interrupted search has only excluded values below (1 - eps) U in the discarded nodes (Prop. 5)
    lb = min([closed_lb, U if not heap else U * (1 - eps)] + [n[0] for n in heap])
    out_terms = [(float(a), float(B[i])) for a, i in zip(best_al, best_key) if a > 0]
    return out_terms, U, {"lb": lb, "gap": max(0.0, (U - lb) / U) if U > 0 else 0.0, "time": time.time() - t0,
                          "nodes": nodes, "n_qp": cache.n_solves, "root_lb": root_lb, "root_ub": hist["root_ub"],
                          "complete": len(heap) == 0}

def fit_with_cuts(fitter, xf, yf, xd, yd, max_rounds=10, add=30, **kw):
    """Constraint generation: fit on (xf,yf); add dominance-violating points from (xd,yd); refit."""
    x, y = xf.copy(), yf.copy(); info_all = []
    for r in range(max_rounds):
        out = fitter(x, y, **kw)
        terms = out[0]
        g = env_eval(terms, xd) if not kw.get("_maxmono") else maxmono_eval(terms, xd)
        viol = yd - g
        bad = np.where(viol > 1e-12)[0]
        info_all.append(len(bad))
        if len(bad) == 0:
            return out, x, y, info_all
        worst = bad[np.argsort(-viol[bad])][:add]
        x = np.concatenate([x, xd[worst]]); y = np.concatenate([y, yd[worst]])
        o = np.argsort(x); x, y = x[o], y[o]
    return out, x, y, info_all

# ----------------------------------------------------------------------------- exact Lagrangian dual (revision)
from scipy.optimize import nnls as _nnls

def _convexified_p1(P, y, groups, N, tol=1e-9):
    """Convex hull relaxation of P1 (the primal form of the Lagrangian dual):
    min ||Pw - y||^2  s.t. Pw >= y, w >= 0, sum_{b in group_j} w_jb <= 1 for each term j.
    groups: list of N index arrays (columns available to each term). Returns value, yhat, mu (KKT multipliers
    of the relaxed curve constraints), w (per column, aggregated) and a flag."""
    if all(np.array_equal(g, groups[0]) for g in groups):      # identical column sets: aggregate the N terms
        groups = [groups[0]]; wmax = float(N)
    else:
        wmax = 1.0
    cols = np.concatenate(groups); owner = np.concatenate([np.full(len(g), j) for j, g in enumerate(groups)])
    Q0 = P[:, cols]; s = np.maximum(Q0.max(axis=0), 1e-300); Q = Q0 / s[None, :]; k = Q.shape[1]
    H = Q.T @ Q; f = Q.T @ y
    fun = lambda v: float(v @ H @ v - 2 * f @ v + y @ y); jac = lambda v: 2 * (H @ v - f)
    cons = [{"type": "ineq", "fun": lambda v: Q @ v - y, "jac": lambda v: Q}]
    for j in range(len(groups)):
        e = np.where(owner == j, 1.0 / s, 0.0)
        cons.append({"type": "ineq", "fun": (lambda v, e=e: wmax - e @ v), "jac": (lambda v, e=e: -e)})
    best = None
    for v0 in [np.full(k, 1.0 / k), np.zeros(k)]:
        r = minimize(fun, v0, jac=jac, bounds=[(0, wmax * sb) for sb in s], constraints=cons, method="SLSQP",
                     options={"ftol": 1e-15, "maxiter": 5000})
        if best is None or r.fun < best.fun: best = r
    v = best.x; yh = Q @ v; res = yh - y
    # KKT multipliers: 2 Q^T res = Q_A^T lam_A - sum_j nu_j e_j/s ... + kappa_Z (all >= 0), solved by NNLS
    A = np.where(res <= tol * np.maximum(1.0, y))[0]
    Z = np.where(v <= 1e-12 * np.maximum(1.0, s))[0]
    blocks = [Q[A].T]
    for j in range(len(groups)): blocks.append(-np.where(owner == j, 1.0 / s, 0.0)[:, None])
    E = np.zeros((k, len(Z))); E[Z, np.arange(len(Z))] = 1.0; blocks.append(E)
    z, _ = _nnls(np.hstack(blocks), 2 * Q.T @ res, maxiter=10 * (k + len(A)))
    lam = np.zeros(len(y)); lam[A] = z[:len(A)]
    mu = lam - 2 * res
    w = np.zeros(P.shape[1]); np.add.at(w, cols, v / s)
    return float(best.fun), mu, w

def _convexified_p2(P, y, groups, N, sbar=1.0):
    if all(np.array_equal(g, groups[0]) for g in groups):
        return _convexified_p2_agg(P, y, groups[0], N, sbar)
    """Convex hull relaxation of P2 as an LP (HiGHS); returns value, multipliers [m1, m2] and w."""
    cols = np.concatenate(groups); owner = np.concatenate([np.full(len(g), j) for j, g in enumerate(groups)])
    Q = P[:, cols]; k = Q.shape[1]; n = len(y)
    c = np.zeros(k + 1); c[-1] = 1.0
    A_up = np.hstack([Q, -np.ones((n, 1))]); A_lo = np.hstack([-Q, np.zeros((n, 1))])
    A_g = np.zeros((len(groups), k + 1))
    for j in range(len(groups)): A_g[j, :k] = (owner == j)
    res = linprog(c, A_ub=np.vstack([A_up, A_lo, A_g]), b_ub=np.concatenate([y, -y, np.ones(len(groups))]),
                  bounds=[(0, 1)] * k + [(0, sbar)], method="highs")
    if res.status == 2: return np.inf, "infeasible", None      # no point with sigma <= sbar: node cannot improve
    if res.status != 0: return np.inf, None, None
    mg = -res.ineqlin.marginals
    mu = np.concatenate([np.maximum(mg[:n], 0), np.maximum(mg[n:2 * n], 0)])
    w = np.zeros(P.shape[1]); np.add.at(w, cols, res.x[:k])
    return float(res.fun), mu, w

def _convexified_p2_agg(P, y, cols, N, sbar):
    Q = P[:, cols]; k = Q.shape[1]; n = len(y)
    c = np.zeros(k + 1); c[-1] = 1.0
    A = np.vstack([np.hstack([Q, -np.ones((n, 1))]), np.hstack([-Q, np.zeros((n, 1))]), np.r_[np.ones(k), 0.0][None, :]])
    res = linprog(c, A_ub=A, b_ub=np.concatenate([y, -y, [float(N)]]), bounds=[(0, N)] * k + [(0, sbar)], method="highs")
    if res.status == 2: return np.inf, "infeasible", None      # no point with sigma <= sbar: node cannot improve
    if res.status != 0: return np.inf, None, None
    mg = -res.ineqlin.marginals
    mu = np.concatenate([np.maximum(mg[:n], 0), np.maximum(mg[n:2 * n], 0)])
    w = np.zeros(P.shape[1]); w[cols] = res.x[:k]
    return float(res.fun), mu, w

def exact_dual_node(Phi, yv, obj, N, iv, U, caps=True, polish=0):
    """Exact Lagrangian bound at a node (term j restricted to grid-index interval iv[j]):
    solve the convexified problem, recover multipliers, evaluate Z_D exactly (valid by weak duality)."""
    groups = [np.arange(l, h + 1) for l, h in iv]
    node = LRNode(Phi, yv, obj, N, obbt=caps)
    if obj == "P1":
        val, mu, w = _convexified_p1(Phi * AMAX, yv, groups, N)
    else:   # the LP carries the same caps as the dual evaluation (exact HiGHS duals)
        acap, _ = node.caps(U)
        val, mu, w = _convexified_p2(Phi * acap[None, :], yv, groups, N, sbar=(U if (caps and np.isfinite(U)) else 1.0))
        if w is not None: w = w * acap
    if isinstance(mu, str):        # P2 LP infeasible with sigma <= U and the cutoff caps (Prop. 4): no solution below U
        return (U if np.isfinite(U) else -np.inf), None, None, val
    if mu is None: return -np.inf, None, w, val
    zd, mu_b, _ = node.run(list(iv), mu, U, max(1, polish))
    return zd, mu_b, w, val

def fit_exact_dual(x, y, B, N, obj, time_limit=600.0, eps=1e-4, enum_leaf=16, rel=False, caps=True):
    """Lagrangian relaxation with the dual solved exactly (convexified primal + exact dual evaluation),
    embedded in a branch-and-bound over exponent-index intervals.  Returns terms, U, info.
    Round-2 corrections: (i) incumbent candidates are ranked by their contribution w_b * max_i x_i^b (not by the raw
    weight w_b); (ii) a leaf (every term restricted to one exponent) is bounded by the exact dual with singleton exponent
    sets, which equals the leaf problem, so that no leaf is discarded on the basis of the coefficient solver alone;
    a leaf whose bound stays below (1-eps) U is kept as an open bound and reported in `lb`."""
    t0 = time.time()
    Phi, yv = _design(x, y, B, rel); nB = len(B); colmax = Phi.max(axis=0)
    cache = AlphaCache(Phi, yv, obj)
    U, best_key, best_al = np.inf, None, None
    def try_key(idx):
        nonlocal U, best_key, best_al
        idx = [int(i) for i in idx]
        if len(set(idx)) == 0: return
        key, (v, al) = cache(idx)
        if v < U: U, best_key, best_al = v, key, al
    mono_terms, _ = fit_monomial(x, y, B, obj, rel)
    if mono_terms: try_key([int(np.argmin(np.abs(B - mono_terms[0][1])))])
    def node_eval(iv):
        zd, mu, w, val = exact_dual_node(Phi, yv, obj, N, iv, U, caps=caps)
        if w is not None:
            contrib = w * colmax                                   # contribution of each exponent to the curve
            order = [b for b in np.argsort(-contrib) if contrib[b] > 1e-12 * max(contrib.max(), 1e-300)]
            if order:
                try_key(order[:N])
                top = order[:N + 2]
                if len(top) > N:
                    for comb in itertools.combinations(top, N): try_key(comb)
        return zd
    root_iv = [(0, nB - 1)] * N
    root_lb = node_eval(root_iv)
    root_lb = max(root_lb, node_eval(root_iv))      # re-evaluate with the improved incumbent (tighter caps)
    root_ub = U
    root_gap = max(0.0, (U - root_lb) / U) if U > 0 else 0.0
    cnt = itertools.count(); heap = [(root_lb, next(cnt), tuple(root_iv))]; nodes = 1; closed = np.inf
    open_leaves = []                                 # bounds of leaves that could not be closed
    while heap:
        if time.time() - t0 > time_limit: break
        lb, _, iv = heapq.heappop(heap)
        if lb >= U * (1 - eps) - 1e-15: closed = min(closed, lb); heap.clear(); break
        sizes = [h - l + 1 for l, h in iv]
        if int(np.prod(sizes, dtype=float)) <= enum_leaf:
            for comb in itertools.product(*[range(l, h + 1) for l, h in iv]):
                if all(comb[j] <= comb[j + 1] for j in range(N - 1)):
                    try_key(comb)
                    zl = max(lb, node_eval([(b, b) for b in comb])); nodes += 1
                    if zl < U * (1 - eps) - 1e-15: open_leaves.append(zl)
            continue
        j = int(np.argmax(sizes)); l, h = iv[j]; m = (l + h) // 2
        left = list(iv); left[j] = (l, m)
        for kk in range(j): left[kk] = (left[kk][0], min(left[kk][1], m))
        right = list(iv); right[j] = (m + 1, h)
        for kk in range(j + 1, N): right[kk] = (max(right[kk][0], m + 1), right[kk][1])
        for ch in (left, right):
            if all(a <= b for a, b in ch):
                zc = max(lb, node_eval(ch)); nodes += 1
                if zc < U * (1 - eps) - 1e-15: heapq.heappush(heap, (zc, next(cnt), tuple(ch)))
    # leaves recorded as open may have been closed by a later, better incumbent
    open_leaves = [z for z in open_leaves if z < U * (1 - eps) - 1e-15]
    # an interrupted search has only excluded values below (1 - eps) U in the discarded nodes (Prop. 5)
    lbf = min([closed, U if not (heap or open_leaves) else U * (1 - eps)] + [hh[0] for hh in heap] + open_leaves)
    out_terms = [(float(a), float(B[i])) for a, i in zip(best_al, best_key) if a > 0]
    return out_terms, U, {"lb": lbf, "gap": max(0.0, (U - lbf) / U) if U > 0 else 0.0, "root_lb": root_lb,
                          "root_ub": root_ub, "root_gap": root_gap, "nodes": nodes, "time": time.time() - t0,
                          "complete": len(heap) == 0 and not open_leaves, "n_open_leaves": len(open_leaves),
                          "n_qp": cache.n_solves}

def fit_milp_p2(x, y, B, N, time_limit=600.0, rel=False, caps=True):
    """Reference MILP for P2 on the grid (HiGHS via scipy.optimize.milp): binary z_b selects exponent b,
    0 <= alpha_b <= abar_b z_b with abar_b the cutoff-based cap from the exact monomial incumbent."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    t0 = time.time()
    Phi, yv = _design(x, y, B, rel); n, m = Phi.shape
    mono, U = fit_monomial(x, y, B, "P2", rel)
    acap = np.minimum(AMAX, np.min((yv + U)[:, None] / Phi, axis=0)) if (caps and np.isfinite(U)) else np.full(m, AMAX)
    # variables: scaled coefficients a'_b = alpha_b / abar_b in [0, 1] (m), z (m), sigma (1); columns scaled by abar_b
    Ps = Phi * acap[None, :]
    c = np.r_[np.zeros(2 * m), 1.0]
    A1 = np.hstack([Ps, np.zeros((n, m)), -np.ones((n, 1))])               # P a' - sigma <= y
    A2 = np.hstack([Ps, np.zeros((n, m)), np.zeros((n, 1))])               # P a' >= y
    A3 = np.hstack([np.eye(m), -np.eye(m), np.zeros((m, 1))])              # a'_b - z_b <= 0
    A4 = np.r_[np.zeros(m), np.ones(m), 0.0][None, :]                      # sum z <= N
    cons = [LinearConstraint(A1, -np.inf, yv), LinearConstraint(A2, yv, np.inf), LinearConstraint(A3, -np.inf, 0.0),
            LinearConstraint(A4, -np.inf, N)]
    integ = np.r_[np.zeros(m), np.ones(m), 0]
    ub = np.r_[np.ones(m), np.ones(m), U if np.isfinite(U) else 1.0]
    res = milp(c, constraints=cons, integrality=integ, bounds=Bounds(np.zeros(2 * m + 1), ub),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-4, "presolve": True})
    el = time.time() - t0
    if res.x is None:
        return None, np.inf, {"time": el, "status": res.status, "complete": False}
    # keep the N exponents selected by z and re-solve their coefficients exactly (removes tolerance artefacts)
    ap = res.x[:m]; sel = np.sort(np.argsort(-ap)[:N]); sel = sel[ap[sel] > 1e-12]
    v, al = solve_alpha(Ps[:, sel], yv, "P2")                              # re-solve on the scaled columns
    if al is None: return None, np.inf, {"time": el, "status": res.status, "complete": False}
    terms = [(float(a_ * acap[b]), float(B[b])) for a_, b in zip(al, sel) if a_ > 0]
    val = objective_any(terms, x, y, "P2", rel)
    return terms, val, {"time": el, "status": res.status, "complete": res.status == 0,
                        "mip_gap": float(getattr(res, "mip_gap", np.nan)), "lb": float(getattr(res, "mip_dual_bound", np.nan))}
