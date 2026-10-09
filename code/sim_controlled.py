"""IJIS v08 (review M5, M6): controlled simulation with independently generated populations.

Two matchers with Gaussian scores whose population error rates are known exactly:
  genuine claim of subject k :  s_m = mu_m + sqrt(VS) a_km + sqrt(1 - VS) e_km
  impostor pair (k, l), k!=l :  s_m = sqrt(VS/2) b_km + sqrt(VS/2) c_lm + sqrt(1 - VS) e_klm
All components are standard normal and correlated across the two matchers with rho_g (genuine) or rho_i (impostor),
so every score has unit variance and the class-conditional correlation between matchers is rho_g or rho_i.  The
subject effects (share VS of the variance) make comparisons that share a subject dependent, and matcher m2 is rounded
to multiples of 0.25 (ties, as with the integer fingerprint scores of BSSR1).  Each replicate draws N training
subjects, fits the proposed envelopes (exact-dual LR-BB, P1, N = 2), designs all four orders by the GP, calibrates
them exactly as in the paper (step (i) Clopper-Pearson stage thresholds, step (ii) subject bootstrap, B = 300,
max(t_CP, t_boot) for the final threshold), selects the order with the smallest training FRR, and evaluates the
deployed thresholds on the POPULATION (bivariate normal probabilities, no test sample):
  far_pop <= alpha     coverage of the population FAR requirement (nominal 0.95 for the bootstrap bound),
  frr_pred >= frr_pop  step-(iii) prediction versus population FRR (Proposition 2 needs rho_g = 0),
  frr_pred >= frr_prod step-(iii) prediction versus the training FRR under the product of the per-matcher
                       training distributions (Lemma of Proposition 1: holds by construction).
A second arm applies the held-out calibration of the paper: design and order on fold A (first N/2 subjects), final
threshold re-set by the bootstrap on fold B (other N/2 subjects) for that design only (not deployed if it fails).
Usage: python sim_controlled.py <N> <rho_g> <rep0> <rep1>  ->  results/E9sim/sim_N<N>_r<rho_g>_<rep0>_<rep1>.csv"""
import json, os, sys, time, numpy as np, pandas as pd
from scipy.stats import norm
from scipy import integrate
import serial_design as sd
from serial_design import gp_design, calibrated_design, simulate, final_threshold_boot
from experiment_core import fit_envelope, all_orders

M = ["m1", "m2"]; MU = {"m1": 4.0, "m2": 4.5}; H = {"m1": 0.0, "m2": 0.25}; VS = 0.3
ALPHAS = [1e-2, 1e-3]; B = 300; CONF = 0.95


def corr_pair(rng, shape, rho):
    z1 = rng.standard_normal(shape); z2 = rng.standard_normal(shape)
    return z1, rho * z1 + np.sqrt(1 - rho ** 2) * z2


def draw(N, rho_g, rho_i, rng):
    S = {}
    a1, a2 = corr_pair(rng, N, rho_g); e1, e2 = corr_pair(rng, N, rho_g)
    b1, b2 = corr_pair(rng, N, rho_i); c1, c2 = corr_pair(rng, N, rho_i); E1, E2 = corr_pair(rng, (N, N), rho_i)
    for m, a, e, b, c, E in [("m1", a1, e1, b1, c1, E1), ("m2", a2, e2, b2, c2, E2)]:
        X = np.sqrt(VS / 2) * b[:, None] + np.sqrt(VS / 2) * c[None, :] + np.sqrt(1 - VS) * E
        X[np.diag_indices(N)] = MU[m] + np.sqrt(VS) * a + np.sqrt(1 - VS) * e
        if H[m] > 0: X = np.round(X / H[m]) * H[m]
        S[m] = X
    return S, np.eye(N, dtype=bool)


def eff(t, m):
    """Continuous threshold equivalent to `score >= t` after rounding of matcher m."""
    if not np.isfinite(t) or H[m] == 0: return t
    return (np.ceil(t / H[m]) - 0.5) * H[m]          # exact for H = 0.25 (a power of two), also for t = nextafter(grid value)


def band(lo, hi, c, mu1, mu2, rho, upper):
    """P(lo <= X1 < hi, X2 >= c) (upper) or P(lo <= X1 < hi, X2 < c), (X1, X2) ~ N((mu1, mu2), [[1, rho], [rho, 1]])."""
    lo, hi = lo - mu1, hi - mu1
    if hi <= lo: return 0.0
    if not np.isfinite(c): return (norm.cdf(hi) - norm.cdf(lo)) * (float(c < 0) if upper else float(c > 0))
    s = np.sqrt(1 - rho ** 2)
    f = (lambda z: norm.pdf(z) * norm.sf((c - mu2 - rho * z) / s)) if upper else (lambda z: norm.pdf(z) * norm.cdf((c - mu2 - rho * z) / s))
    lo_, hi_ = max(lo, -12.0), min(hi, 12.0)
    if hi_ <= lo_: return 0.0
    v, _ = integrate.quad(f, lo_, hi_, epsabs=1e-14, epsrel=1e-10, limit=200)
    return v


def pop_rates(order, thr, rho_g, rho_i):
    if len(order) == 1:
        t = eff(thr[0], order[0]); return float(norm.sf(t)), float(norm.cdf(t - MU[order[0]]))
    m1, m2 = order; ta, tr = eff(thr[0][0], m1), eff(thr[0][1], m1); t2 = eff(thr[1], m2)
    far = norm.sf(ta) + band(tr, ta, t2, 0.0, 0.0, rho_i, True)
    frr = norm.cdf(tr - MU[m1]) + band(tr, ta, t2, MU[m1], MU[m2], rho_g, False)
    return float(far), float(frr)


def marg(sc, G, t):
    """Training FAR / FRR of one matcher at threshold t (accept iff score >= t)."""
    return float((sc[~G] >= t).mean()), float((sc[G] < t).mean())


def designs(Sc, G, boot_seed, rep, alpha, envs):
    """All calibrated designs (four orders) on the data Sc, as in the paper; returns a list of dicts."""
    n = G.shape[0]; subj = np.arange(n); sd.BOOT.update(row_subj=subj.copy(), col_subj=subj.copy(), B=B, seed=boot_seed)
    IMPS = {m: np.sort(Sc[m][~G]) for m in M}; out = []
    for order in all_orders(M):
        ev = [envs[m] for m in order]; sc = [Sc[m] for m in order]; d = gp_design(ev, alpha)
        if d is None: continue
        imps = [IMPS[m] for m in order]
        c = calibrated_design(d, ev, sc, G, alpha, conf=CONF, joint=True, imp_sorted=imps, final="boot")
        if c is None: continue
        pre = calibrated_design(d, ev, sc, G, alpha, conf=CONF, joint=False, imp_sorted=imps)[0]       # step (i) only
        thr, pred = c; fa, fr, _, _ = simulate(sc, G, thr)
        if len(order) == 1: fprod = marg(Sc[order[0]], G, thr[0])[1]
        else:
            ra, rr = marg(Sc[order[0]], G, thr[0][0])[1], marg(Sc[order[0]], G, thr[0][1])[1]
            fprod = rr + (ra - rr) * marg(Sc[order[1]], G, thr[1])[1]
        out.append(dict(order=order, thr=thr, pre=pre, far_train=fa, frr_train=fr, frr_prod=fprod, frr_pred=pred))
    return out


def jt(thr):
    return json.dumps([list(t) if isinstance(t, tuple) else t for t in thr])


def main():
    N, rho_g, r0, r1 = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); rho_i = rho_g / 3
    rows = []; t0 = time.time(); h = N // 2
    for rep in range(r0, r1):
        rng = np.random.default_rng(100000 * int(round(rho_g * 10)) + 1000 * (N // 100) + rep)
        Sc, G = draw(N, rho_g, rho_i, rng)
        envs = {m: fit_envelope("XD-P1-N2", Sc[m][G], Sc[m][~G], seed=rep, time_limit=300)[0] for m in M}
        # held-out calibration: design fold A (first half of the subjects), calibration fold B (second half)
        ScA = {m: Sc[m][:h, :h] for m in M}; ScB = {m: Sc[m][h:, h:] for m in M}; GA = np.eye(h, dtype=bool); GB = np.eye(N - h, dtype=bool)
        envA = {m: fit_envelope("XD-P1-N2", ScA[m][GA], ScA[m][~GA], seed=rep, time_limit=300)[0] for m in M}
        base = dict(N=N, rho_g=rho_g, rho_i=rho_i, rep=rep)
        for alpha in ALPHAS:
            # (a) bootstrap calibration on all N training subjects, order with the smallest training FRR
            des = designs(Sc, G, 7 + rep, rep, alpha, envs)
            if not des: rows.append(dict(base, alpha=alpha, calib="boot", feasible=False))
            mn = min([d_["frr_train"] for d_ in des], default=np.nan)
            for d_ in des:
                pf, pr = pop_rates(d_["order"], d_["thr"], rho_g, rho_i)
                rows.append(dict(base, alpha=alpha, calib="boot", order=">".join(d_["order"]), n_stages=len(d_["order"]), feasible=True,
                                 selected=bool(abs(d_["frr_train"] - mn) <= 1e-12), deployed=True, far_train=d_["far_train"],
                                 frr_train=d_["frr_train"], frr_prod=d_["frr_prod"], frr_pred=d_["frr_pred"], far_pop=pf, frr_pop=pr, thr=jt(d_["thr"])))
            # (b) held-out: order fixed on fold A (fold-A calibrated FRR), final threshold re-set on fold B for that design only
            desA = designs(ScA, GA, 3000 + rep, rep, alpha, envA)
            if not desA: rows.append(dict(base, alpha=alpha, calib="xfit", feasible=False)); continue
            mnA = min(d_["frr_train"] for d_ in desA)
            for d_ in desA:
                if abs(d_["frr_train"] - mnA) > 1e-12: continue
                subj = np.arange(N - h); sd.BOOT.update(row_subj=subj.copy(), col_subj=subj.copy(), B=B, seed=4000 + rep)
                scB = [ScB[m] for m in d_["order"]]; pre = d_["thr"][:-1]
                t = final_threshold_boot(scB, GB, pre, alpha, CONF)
                row = dict(base, alpha=alpha, calib="xfit", order=">".join(d_["order"]), n_stages=len(d_["order"]), feasible=True, selected=True)
                if t is None: row["deployed"] = False; rows.append(row); continue
                thr = list(pre) + [max(d_["pre"][-1], t)]
                faB, frB, _, _ = simulate(scB, GB, thr); pf, pr = pop_rates(d_["order"], thr, rho_g, rho_i)
                row.update(deployed=True, far_train=faB, frr_train=frB, far_pop=pf, frr_pop=pr, thr=jt(thr)); rows.append(row)
        print(f"[sim N={N} rho={rho_g}] rep {rep} {time.time() - t0:.0f}s", flush=True)
    os.makedirs("../results/E9sim", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"../results/E9sim/sim_N{N}_r{rho_g:.1f}_{r0:03d}_{r1:03d}.csv", index=False)


if __name__ == "__main__":
    main()
