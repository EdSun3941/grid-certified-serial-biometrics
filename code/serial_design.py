"""Serial dual-threshold design: GP model (TDSC-type) with envelope constraints, threshold
recovery, joint simulation, and classical serial-rule baselines.

Scores are similarities (accept iff s >= t).  Stage s<S has acceptance threshold t_acc and
rejection threshold t_rej (t_rej <= t_acc); scores in [t_rej, t_acc) go to the next stage.
Rates at a threshold: a = FAR(t) = P(imp >= t), r = FRR(t) = P(gen < t).
"""
import numpy as np
from scipy.optimize import minimize

LOGMIN = np.log(1e-9)

# ----------------------------------------------------------------------------- envelopes
class Envelope:
    def __init__(self, kind, terms, xmin, xmax):
        self.kind, self.terms, self.xmin, self.xmax = kind, terms, xmin, xmax   # kind: 'posy'|'max'
    def __call__(self, a):
        a = np.asarray(a, float)
        if self.kind == "max":
            return np.max([al * a ** be for al, be in self.terms], axis=0)
        return sum(al * a ** be for al, be in self.terms)
    def gp_rows(self, ia, ir, n):
        """Posynomial constraints g(a)/r <= 1 as list of (logc, E) blocks."""
        blocks = []
        pieces = [self.terms] if self.kind == "posy" else [[t] for t in self.terms]
        for p in pieces:
            logc = np.array([np.log(al) for al, be in p]); E = np.zeros((len(p), n))
            for k, (al, be) in enumerate(p): E[k, ia] = be; E[k, ir] = -1.0
            blocks.append((logc, E))
        return blocks

# ----------------------------------------------------------------------------- GP
def _lse(logc, E, z):
    v = logc + E @ z; m = v.max(); w = np.exp(v - m); s = w.sum()
    return m + np.log(s), (w / s) @ E

def gp_design(envs, alpha, starts=(0.5, 0.1, 0.02), imp_stage_cap=None, aacc_max=None):
    """min system FRR s.t. system FAR <= alpha (TDSC composition, posynomial).
    Optional design constraints (round-2 revision): imp_stage_cap = kappa adds the posynomial bound
    1 + sum_{s<S} prod_{j<=s} a_rej_j <= kappa on the expected number of stages of an impostor claim; aacc_max[s]
    caps the stage FAR at the accept threshold of stage s (e.g. to limit acceptance of a spoof of that modality).
    Returns dict with rates per stage, objective and FAR model value, or None if infeasible."""
    S = len(envs)
    # variable layout
    idx = []; n = 0
    for s in range(S):
        if s < S - 1: idx.append({"aa": n, "ar": n + 1, "ra": n + 2, "rr": n + 3}); n += 4
        else: idx.append({"a": n, "r": n + 1}); n += 2
    def posy_frr():
        rows = []
        for s in range(S):
            e = np.zeros(n)
            for j in range(s): e[idx[j]["ra"]] = 1
            e[idx[s]["rr"] if s < S - 1 else idx[s]["r"]] = 1; rows.append(e)
        return np.zeros(len(rows)), np.array(rows)
    def posy_far():
        rows = []
        for s in range(S):
            e = np.zeros(n)
            for j in range(s): e[idx[j]["ar"]] = 1
            e[idx[s]["aa"] if s < S - 1 else idx[s]["a"]] = 1; rows.append(e)
        return np.zeros(len(rows)), np.array(rows)
    obj = posy_frr(); far = posy_far()
    cons = [{"type": "ineq", "fun": lambda z: np.log(alpha) - _lse(*far, z)[0],
             "jac": lambda z: -_lse(*far, z)[1]}]
    bounds = [None] * n
    for s in range(S):
        env = envs[s]; lo, hi = np.log(env.xmin), np.log(env.xmax)
        pairs = [(idx[s]["aa"], idx[s]["ra"]), (idx[s]["ar"], idx[s]["rr"])] if s < S - 1 else [(idx[s]["a"], idx[s]["r"])]
        for ia, ir in pairs:
            bounds[ia] = (lo, hi); bounds[ir] = (LOGMIN, 0.0)
            for logc, E in env.gp_rows(ia, ir, n):
                cons.append({"type": "ineq", "fun": (lambda z, l=logc, E=E: -_lse(l, E, z)[0]),
                             "jac": (lambda z, l=logc, E=E: -_lse(l, E, z)[1])})
        if aacc_max is not None and aacc_max[s] is not None:
            ia = idx[s]["aa"] if s < S - 1 else idx[s]["a"]
            bounds[ia] = (bounds[ia][0], min(bounds[ia][1], float(np.log(max(aacc_max[s], env.xmin)))))
        if s < S - 1:
            i = idx[s]
            e1 = np.zeros(n); e1[i["ar"]] = 1; e1[i["aa"]] = -1
            e2 = np.zeros(n); e2[i["ra"]] = 1; e2[i["rr"]] = -1
            cons.append({"type": "ineq", "fun": (lambda z, e=e1: e @ z), "jac": (lambda z, e=e1: e)})
            cons.append({"type": "ineq", "fun": (lambda z, e=e2: e @ z), "jac": (lambda z, e=e2: e)})
    if imp_stage_cap is not None and S > 1:
        E = np.zeros((S, n)); logc = np.full(S, -np.log(imp_stage_cap))
        for s_ in range(1, S):
            for j in range(s_): E[s_, idx[j]["ar"]] = 1
        cons.append({"type": "ineq", "fun": (lambda z, l=logc, E=E: -_lse(l, E, z)[0]),
                     "jac": (lambda z, l=logc, E=E: -_lse(l, E, z)[1])})
    best = None
    for arej in starts:
        z0 = np.zeros(n)
        for s in range(S):
            env = envs[s]; i = idx[s]
            clip = lambda a: float(np.clip(a, env.xmin, env.xmax))
            if s < S - 1:
                aa, ar = clip(alpha / (2 * S)), clip(arej)
                if ar < aa: ar = aa
                z0[i["aa"]], z0[i["ar"]] = np.log(aa), np.log(ar)
                z0[i["ra"]] = np.log(min(1.0, env(aa) * 1.01)); z0[i["rr"]] = np.log(min(1.0, env(ar) * 1.01))
                if z0[i["rr"]] > z0[i["ra"]]: z0[i["rr"]] = z0[i["ra"]]
            else:
                a = clip(alpha / 2); z0[i["a"]] = np.log(a); z0[i["r"]] = np.log(min(1.0, env(a) * 1.01))
        try:
            res = minimize(lambda z: _lse(*obj, z)[0], z0, jac=lambda z: _lse(*obj, z)[1], bounds=bounds,
                           constraints=cons, method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000})
        except Exception:
            continue
        z = res.x
        viol = min(c["fun"](z) for c in cons)
        if viol < -1e-7: continue
        val = _lse(*obj, z)[0]
        if best is None or val < best[0]: best = (val, z)
    if best is None:
        return None
    z = best[1]; v = np.exp(z)
    stages = []
    for s in range(S):
        i = idx[s]
        if s < S - 1: stages.append({"a_acc": v[i["aa"]], "a_rej": v[i["ar"]], "r_acc": v[i["ra"]], "r_rej": v[i["rr"]]})
        else: stages.append({"a_acc": v[i["a"]], "r_acc": v[i["r"]]})
    return {"stages": stages, "frr_pred": float(np.exp(best[0])), "far_model": float(np.exp(_lse(*far, z)[0]))}

def composition(stages, key_acc, key_rej, exact=True):
    """System rate from per-stage rates. exact=True uses pass-through (acc - rej)."""
    tot, pas = 0.0, 1.0; S = len(stages)
    for s, st in enumerate(stages):
        if s < S - 1:
            if key_acc == "a_acc":   # FAR: accepted at a_acc, pass-through a_rej - a_acc
                tot += pas * st["a_acc"]; pas *= (st["a_rej"] - st["a_acc"]) if exact else st["a_rej"]
            else:                    # FRR: rejected at r_rej, pass-through r_acc - r_rej
                tot += pas * st["r_rej"]; pas *= (st["r_acc"] - st["r_rej"]) if exact else st["r_acc"]
        else:
            tot += pas * (st["a_acc"] if key_acc == "a_acc" else st["r_acc"])
    return tot

# ----------------------------------------------------------------------------- thresholds & simulation
def thr_for_far(imp_sorted, a):
    """Smallest threshold t with FAR(t)=#(imp>=t)/n <= a."""
    n = len(imp_sorted); k = int(np.floor(a * n + 1e-9))
    if k >= n: return -np.inf
    return float(np.nextafter(imp_sorted[n - k - 1], np.inf))

def thr_for_frr(gen_sorted, r):
    """Largest threshold t with FRR(t)=#(gen<t)/n <= r."""
    n = len(gen_sorted); k = int(np.floor(r * n + 1e-9))
    if k >= n: return np.inf
    return float(gen_sorted[k])

def rates_at(gen_sorted, imp_sorted, t):
    far = 1.0 - np.searchsorted(imp_sorted, t, side="left") / len(imp_sorted)
    frr = np.searchsorted(gen_sorted, t, side="left") / len(gen_sorted)
    return float(far), float(frr)

def simulate(scores, genuine, thr):
    """scores: list of 2-D arrays (one per stage, same shape), genuine mask, thr: list of
    (t_acc, t_rej) for s<S and t for the final stage.  Returns FAR, FRR, mean stages (gen, imp)."""
    S = len(scores); decided = np.zeros(genuine.shape, bool); acc = np.zeros(genuine.shape, bool)
    used = np.zeros(genuine.shape, np.int8)
    for s in range(S):
        und = ~decided; used[und] += 1
        if s < S - 1:
            ta, tr = thr[s]
            a = und & (scores[s] >= ta); r = und & (scores[s] < tr)
            acc |= a; decided |= a | r
        else:
            acc |= und & (scores[s] >= thr[s]); decided[:] = True
    g = genuine
    return float(acc[~g].mean()), float((~acc[g]).mean()), float(used[g].mean()), float(used[~g].mean())

def recover_thresholds(design, imp_sorted_list):
    thr = []; S = len(design["stages"])
    for s, st in enumerate(design["stages"]):
        im = imp_sorted_list[s]
        if s < S - 1: thr.append((thr_for_far(im, st["a_acc"]), thr_for_far(im, st["a_rej"])))
        else: thr.append(thr_for_far(im, st["a_acc"]))
    return thr

def empirical_stage_rates(thr, gen_sorted_list, imp_sorted_list):
    out = []; S = len(thr)
    for s in range(S):
        gs, im = gen_sorted_list[s], imp_sorted_list[s]
        if s < S - 1:
            fa, ra = rates_at(gs, im, thr[s][0]); fr, rr = rates_at(gs, im, thr[s][1])
            out.append({"a_acc": fa, "a_rej": fr, "r_acc": ra, "r_rej": rr})
        else:
            fa, ra = rates_at(gs, im, thr[s]); out.append({"a_acc": fa, "r_acc": ra})
    return out

# ----------------------------------------------------------------------------- classical baselines
def _final_threshold(scores, genuine, pre_thr, alpha):
    """Given thresholds for stages < S, choose the smallest final threshold with train FAR <= alpha."""
    S = len(scores); decided = np.zeros(genuine.shape, bool); acc = np.zeros(genuine.shape, bool)
    for s in range(S - 1):
        ta, tr = pre_thr[s]; und = ~decided
        a = und & (scores[s] >= ta); r = und & (scores[s] < tr); acc |= a; decided |= a | r
    imp = ~genuine; n_imp = imp.sum(); fa0 = (acc & imp).sum()
    budget = int(np.floor(alpha * n_imp + 1e-9)) - fa0
    if budget < 0: return None
    reach = scores[-1][(~decided) & imp]
    if len(reach) == 0: return -np.inf
    rs = np.sort(reach)
    if budget >= len(rs): return -np.inf
    return float(np.nextafter(rs[len(rs) - budget - 1], np.inf))

def _final(scores, genuine, pre, alpha, conf):
    return _final_threshold(scores, genuine, pre, alpha) if conf is None else final_threshold_conf(scores, genuine, pre, alpha, conf)

def rule_marcialis(scores, genuine, alpha, conf=None):
    pre = []
    for s in range(len(scores) - 1):
        sc = scores[s]; pre.append((float(np.nextafter(sc[~genuine].max(), np.inf)), float(sc[genuine].min())))
    tS = _final(scores, genuine, pre, alpha, conf)
    return None if tS is None else pre + [tS]

def rule_symmetric(scores, genuine, alpha, grid=(0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3), conf=None):
    from data import roc
    best = None
    for p in grid:
        pre = []
        for s in range(len(scores) - 1):
            sc = scores[s]; gen, imp = np.sort(sc[genuine]), np.sort(sc[~genuine])
            t, far, frr = roc(gen, imp); e = int(np.argmin(np.abs(far - frr)))
            ta = thr_for_far(imp, max(far[e] - p, 0.0)); tr = thr_for_frr(gen, max(frr[e] - p, 0.0))
            pre.append((ta, min(tr, ta)))
        tS = _final(scores, genuine, pre, alpha, conf)
        if tS is None: continue
        thr = pre + [tS]; far_, frr_, _, _ = simulate(scores, genuine, thr)
        if best is None or frr_ < best[0]: best = (frr_, thr, p)
    return None if best is None else best[1]

def rule_direct(scores, genuine, alpha, xmin, n_grid=20, conf=None):
    """Non-conservative direct empirical search over stage-1 thresholds (2-stage orders only)."""
    assert len(scores) == 2
    sc = scores[0]; gen, imp = np.sort(sc[genuine]), np.sort(sc[~genuine])
    a_acc = np.unique(np.concatenate([np.geomspace(max(xmin, 1e-7), alpha, n_grid), [0.0]]))
    a_rej = np.geomspace(max(alpha, xmin), 0.95, n_grid)
    best = None
    for aa in a_acc:
        ta = thr_for_far(imp, aa)
        for ar in a_rej:
            tr = min(thr_for_far(imp, ar), ta)
            tS = _final(scores, genuine, [(ta, tr)], alpha, conf)
            if tS is None: continue
            thr = [(ta, tr), tS]; far_, frr_, _, _ = simulate(scores, genuine, thr)
            if best is None or frr_ < best[0]: best = (frr_, thr)
    return None if best is None else best[1]

def parallel_sum(scores_tr, gen_tr, scores_te, alpha, conf=None):
    """z-normalised sum rule over all given matchers (impostor mean/std from training)."""
    z_tr = 0; z_te = 0
    for a, b in zip(scores_tr, scores_te):
        mu, sd = a[~gen_tr].mean(), a[~gen_tr].std() + 1e-12
        z_tr = z_tr + (a - mu) / sd; z_te = z_te + (b - mu) / sd
    imp = np.sort(z_tr[~gen_tr]); t = thr_for_far(imp, alpha) if conf is None else thr_for_far_conf(imp, alpha, conf)
    if t is None: t = float(np.nextafter(imp[-1], np.inf))
    return t, z_tr, z_te

# ----------------------------------------------------------------------------- SPRT (supplementary)
def llr_function(gen, imp, nbins=512):
    """Log-likelihood ratio log p_gen(s)/p_imp(s) from smoothed histograms (Gaussian-smoothed,
    Silverman bandwidth), returned as a vectorised interpolant."""
    lo, hi = min(gen.min(), imp.min()), max(gen.max(), imp.max())
    edges = np.linspace(lo, hi, nbins + 1); c = 0.5 * (edges[1:] + edges[:-1]); w = edges[1] - edges[0]
    def dens(v):
        h, _ = np.histogram(v, bins=edges); h = h.astype(float)
        bw = 1.06 * v.std() * len(v) ** (-0.2); k = max(1, int(round(bw / w)))
        ker = np.exp(-0.5 * (np.arange(-4 * k, 4 * k + 1) / k) ** 2); ker /= ker.sum()
        p = np.convolve(h, ker, mode="same") + 1e-3   # floor avoids log(0)
        return p / p.sum()
    l = np.log(dens(gen)) - np.log(dens(imp))
    return lambda s: np.interp(s, c, l)

def rule_sprt(scores, genuine, alpha, n_grid=12, conf=None):
    """Wald-type sequential test on cumulative LLR with constant (A, B); final stage threshold tau
    chosen so that train FAR <= alpha.  Returns (thresholds on cumulative-LLR scores, cum scores fn)."""
    S = len(scores); llrs = [llr_function(sc[genuine], sc[~genuine]) for sc in scores]
    cum = []; acc = 0
    for s in range(S):
        acc = acc + llrs[s](scores[s]); cum.append(acc)
    lg, li = cum[0][genuine], cum[0][~genuine]
    A_grid = np.quantile(li, 1 - np.geomspace(max(1.0 / li.size, alpha * 1e-2), alpha, n_grid))
    B_grid = np.quantile(lg, np.geomspace(1e-3, 0.3, n_grid))
    best = None
    for A in A_grid:
        for Bt in B_grid:
            if Bt >= A: continue
            pre = [(float(A), float(Bt))] * (S - 1)
            tS = _final(cum, genuine, pre, alpha, conf)
            if tS is None: continue
            thr = pre + [tS]; far_, frr_, _, _ = simulate(cum, genuine, thr)
            if best is None or frr_ < best[0]: best = (frr_, thr)
    return None if best is None else (best[1], llrs)

# ----------------------------------------------------------------------------- calibrated deployment
from scipy.stats import beta as _beta

def far_budget(n, a, conf=None):
    """Largest number k of false accepts among n impostor trials such that the estimate is
    acceptable: conf=None -> k/n <= a;  conf=c -> one-sided Clopper-Pearson upper bound <= a."""
    kmax = int(np.floor(a * n + 1e-9))
    if conf is None or kmax < 0: return kmax
    k = kmax
    while k >= 0 and _beta.ppf(conf, k + 1, n - k) > a:
        k -= 1
    return k          # may be -1: even zero false accepts cannot certify a

def thr_for_far_conf(imp_sorted, a, conf=None):
    n = len(imp_sorted); k = far_budget(n, a, conf)
    if k < 0: return None
    if k >= n: return -np.inf
    return float(np.nextafter(imp_sorted[n - k - 1], np.inf))

def final_threshold_conf(scores, genuine, pre_thr, alpha, conf=None):
    S = len(scores); decided = np.zeros(genuine.shape, bool); acc = np.zeros(genuine.shape, bool)
    for s in range(S - 1):
        ta, tr = pre_thr[s]; und = ~decided
        a = und & (scores[s] >= ta); r = und & (scores[s] < tr); acc |= a; decided |= a | r
    imp = ~genuine; n_imp = int(imp.sum()); fa0 = int((acc & imp).sum())
    budget = far_budget(n_imp, alpha, conf) - fa0
    if budget < 0: return None
    reach = np.sort(scores[-1][(~decided) & imp])
    if budget >= len(reach): return -np.inf
    return float(np.nextafter(reach[len(reach) - budget - 1], np.inf))

def calibrated_design(design, envs, tr_scores, G, alpha, conf=0.95, joint=True, imp_sorted=None, final="cp"):
    """Deploy a GP design safely:
    (1) each stage threshold is set so that the one-sided CP upper bound of its training FAR is <= the
        GP rate (statistical margin);
    (2) joint=True: the final-stage threshold is raised, if needed, until the CP upper bound of the
        JOINT training FAR (serial simulation, no independence assumption) is <= alpha.
    Returns thresholds and the conservative FRR prediction re-evaluated with the envelope at the
    realised stage FARs (exact composition), or None if no certified design exists."""
    S = len(design["stages"]); thr = []
    if imp_sorted is None: imp_sorted = [np.sort(sc[~G]) for sc in tr_scores]
    for s, st in enumerate(design["stages"]):
        im = imp_sorted[s]
        if s < S - 1:
            ta = thr_for_far_conf(im, st["a_acc"], conf); tr_ = thr_for_far_conf(im, st["a_rej"], conf)
            if ta is None: ta = float(np.nextafter(im[-1], np.inf))
            if tr_ is None: tr_ = ta
            thr.append((ta, min(tr_, ta)))
        else:
            t = thr_for_far_conf(im, st["a_acc"], conf)
            thr.append(float(np.nextafter(im[-1], np.inf)) if t is None else t)
    if joint:
        tS = (final_threshold_boot(tr_scores, G, thr[:-1], alpha, conf) if final == "boot"
              else final_threshold_conf(tr_scores, G, thr[:-1], alpha, conf))
        if tS is None: return None
        thr[-1] = max(thr[-1], tS)
    # conservative FRR prediction at the realised (training) stage FARs
    stages = []
    for s in range(S):
        im = imp_sorted[s]; n = len(im)
        def a_at(t): return float(1.0 - np.searchsorted(im, t, side="left") / n)
        if s < S - 1:
            aa, ar = a_at(thr[s][0]), a_at(thr[s][1])
            ra = float(envs[s](max(aa, envs[s].xmin))); rr = float(envs[s](max(ar, envs[s].xmin)))
            stages.append({"a_acc": aa, "a_rej": ar, "r_acc": min(ra, 1.0), "r_rej": min(rr, ra, 1.0)})
        else:
            aa = a_at(thr[s]); stages.append({"a_acc": aa, "r_acc": min(float(envs[s](max(aa, envs[s].xmin))), 1.0)})
    return thr, composition(stages, "r_acc", "r_rej", exact=True)

# ----------------------------------------------------------------------------- subject-level calibration (revision)
# Two-way subject bootstrap of the JOINT training FAR: every resampled subject brings all of its comparisons
# (as probe and as reference), so the dependence between comparisons that share a subject is preserved
# (cf. the subsets bootstrap of Bolle et al.).  BOOT is set by the calling script for each split.
BOOT = {"row_subj": None, "col_subj": None, "B": 300, "seed": 0}

_BW_CACHE = {}
def _boot_weights(B, seed):
    key = (id(BOOT["row_subj"]), id(BOOT["col_subj"]), B, seed)
    if key in _BW_CACHE: return _BW_CACHE[key]
    _BW_CACHE.clear()
    rs, cs = BOOT["row_subj"], BOOT["col_subj"]
    subj = np.unique(np.concatenate([rs, cs])); idx = {s: k for k, s in enumerate(subj)}
    ri = np.array([idx[s] for s in rs]); ci = np.array([idx[s] for s in cs])
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(subj), np.full(len(subj), 1.0 / len(subj)), size=B).astype(np.float64)   # B x n_subj
    _BW_CACHE[key] = (W[:, ri], W[:, ci])                              # row and column weights, B x n_r, B x n_c
    return _BW_CACHE[key]

def final_threshold_boot(scores, genuine, pre_thr, alpha, conf=0.95, B=None, seed=None):
    """Smallest final threshold such that the conf-quantile of the bootstrap distribution of the joint
    training FAR (earlier stages fixed) does not exceed alpha.  Returns None if even rejecting every
    remaining claim cannot meet alpha."""
    B = B or BOOT["B"]; seed = BOOT["seed"] if seed is None else seed
    S = len(scores); decided = np.zeros(genuine.shape, bool); acc = np.zeros(genuine.shape, bool)
    for s in range(S - 1):
        ta, tr = pre_thr[s]; und = ~decided
        a = und & (scores[s] >= ta); r = und & (scores[s] < tr); acc |= a; decided |= a | r
    imp = ~genuine; n_imp = int(imp.sum())
    wr, wc = _boot_weights(B, seed)
    gi, gj = np.nonzero(genuine)
    den = wr.sum(1) * wc.sum(1) - (wr[:, gi] * wc[:, gj]).sum(1)                       # bootstrap impostor counts
    ai, aj = np.nonzero(acc & imp)
    base = (wr[:, ai] * wc[:, aj]).sum(1) if len(ai) else np.zeros(B)
    if np.quantile(base / den, conf) > alpha: return None
    pi, pj = np.nonzero((~decided) & imp)
    if len(pi) == 0: return -np.inf
    sc = scores[-1][pi, pj]
    K = min(len(sc), int(3 * alpha * n_imp) + 200)
    top = np.argpartition(-sc, K - 1)[:K] if K < len(sc) else np.arange(len(sc))
    top = top[np.argsort(-sc[top], kind="stable")]; st = sc[top]
    # evaluate only at tie-group ends (all pairs with an equal score are accepted together)
    ends = np.r_[np.nonzero(np.diff(st) != 0)[0], len(st) - 1]
    best_k = -1
    CH = 20000
    run = base.copy()
    for c0 in range(0, len(st), CH):
        sl = top[c0:c0 + CH]
        cum = run[:, None] + np.cumsum(wr[:, pi[sl]] * wc[:, pj[sl]], axis=1)
        q = np.quantile(cum / den[:, None], conf, axis=0)
        e = ends[(ends >= c0) & (ends < c0 + len(sl))]
        ok = e[q[e - c0] <= alpha]
        bad = e[q[e - c0] > alpha]
        if len(ok): best_k = max(best_k, int(ok.max()))
        if len(bad): break                                # q is non-decreasing in k
        run = cum[:, -1]
    if best_k < 0:
        return float(np.nextafter(st[0], np.inf))         # reject all remaining impostors
    if best_k == len(sc) - 1: return -np.inf
    return float(st[best_k])                               # accept iff score >= t (whole tie group included)

def rule_direct_cd(scores, genuine, alpha, xmin, n_grid=10, sweeps=2):
    """Non-conservative direct empirical search for chains of any length: coordinate descent over the
    (accept, reject) FAR budgets of every non-final stage; final threshold at the training point estimate."""
    S = len(scores)
    imps = [np.sort(sc[~genuine]) for sc in scores]
    g_acc = np.unique(np.concatenate([np.geomspace(max(xmin, 1e-7), alpha, n_grid), [0.0]]))
    g_rej = np.geomspace(max(alpha, xmin), 0.95, n_grid)
    cur = [(alpha / S, 0.5)] * (S - 1)
    def evaluate(budgets):
        pre = []
        for s, (aa, ar) in enumerate(budgets):
            ta = thr_for_far(imps[s], aa); tr = min(thr_for_far(imps[s], ar), ta); pre.append((ta, tr))
        tS = _final_threshold(scores, genuine, pre, alpha)
        if tS is None: return np.inf, None
        thr = pre + [tS]; _, frr, _, _ = simulate(scores, genuine, thr); return frr, thr
    best_v, best_thr = evaluate(cur)
    for _ in range(sweeps):
        for s in range(S - 1):
            for aa in g_acc:
                for ar in g_rej:
                    cand = list(cur); cand[s] = (aa, ar); v, thr = evaluate(cand)
                    if v < best_v: best_v, best_thr, cur = v, thr, cand
    return best_thr
