"""Core experiment routines shared by all experiments (fitting, GP design, evaluation)."""
import time, itertools, numpy as np
from data import roc, staircase_points, thin, split_subjects
from envelope_solvers import (fit_hyperbola, fit_monomial, fit_lr, fit_exact_dual, fit_enum, fit_nls_shift, fit_de,
                              fit_maxmono, env_eval, maxmono_eval, objective)
from serial_design import (Envelope, gp_design, composition, recover_thresholds, simulate,
                           empirical_stage_rates, rule_marcialis, rule_symmetric, rule_direct, parallel_sum)

B_DEFAULT = np.round(np.arange(-3, 1e-9, 0.02), 2)

def fitting_data(gen, imp, mode="corner", region=None, nbins=120):
    _, far, frr = roc(gen, imp)
    x, y = staircase_points(far, frr, mode)
    if region is not None:
        m = (x >= region[0]) & (x <= region[1]); x, y = x[m], y[m]
    xf, yf = thin(x, y, nbins)
    return xf, yf, x, y

def _fit_one(method, x, y, B, seed, time_limit):
    t0 = time.time(); info = {}
    rel = method.endswith("-rel")
    if rel: method = method[:-4]
    if method == "HYP":
        terms = fit_hyperbola(x, y); kind = "posy"
    elif method == "MONO":
        terms, _ = fit_monomial(x, y, B, "P1", rel=rel); kind = "posy"
    elif method == "MONO-cont":        # continuous exponent: grid spacing 1e-4 on [-3, 0], closed-form alpha
        terms, _ = fit_monomial(x, y, np.round(np.arange(-3, 1e-9, 1e-4), 4), "P1", rel=rel); kind = "posy"
    elif method.startswith("XD-"):     # exact-dual LR-BB of Sec. IV (round-2 revision)
        _, obj, n = method.split("-"); N = int(n[1:])
        terms, v, info = fit_exact_dual(x, y, B, N, obj, time_limit=time_limit, rel=rel); kind = "posy"
    elif method.startswith("LR-") or method.startswith("LRroot-"):
        _, obj, n = method.split("-"); N = int(n[1:])
        mode = "root" if method.startswith("LRroot") else "bb"
        obbt = not method.startswith("LRroot")
        terms, v, info = fit_lr(x, y, B, N, obj, mode=mode, obbt=obbt, time_limit=time_limit, rel=rel); kind = "posy"
    elif method.startswith("ENUM-"):
        _, obj, n = method.split("-"); terms, v, info = fit_enum(x, y, B, int(n[1:]), obj, time_limit=time_limit, rel=rel); kind = "posy"
    elif method == "NLS-N2":
        terms, _ = fit_nls_shift(x, y, 2, "P1", seed=seed, rel=rel); kind = "posy"
    elif method == "DE-N2":
        terms, _, info = fit_de(x, y, 2, "P1", seed=seed, rel=rel); kind = "posy"
    elif method == "MAXMONO-K3":
        terms, _ = fit_maxmono(x, y, B, "P1", K=3, rel=rel); kind = "max"
    else:
        raise ValueError(method)
    info = dict(info); info["fit_time"] = time.time() - t0
    return kind, terms, info

def fit_envelope(method, gen, imp, B=B_DEFAULT, mode="corner", region=None, nbins=120, seed=0,
                 time_limit=300, max_rounds=10):
    """Fit with constraint generation so that the envelope dominates ALL staircase points in range."""
    xf, yf, xd, yd = fitting_data(gen, imp, mode, region, nbins)
    x, y = xf.copy(), yf.copy(); rounds = 0; t0 = time.time()
    while True:
        kind, terms, info = _fit_one(method, x, y, B, seed, time_limit)
        g = maxmono_eval(terms, xd) if kind == "max" else env_eval(terms, xd)
        viol = yd - g; bad = np.where(viol > 1e-12)[0]
        if len(bad) == 0 or rounds >= max_rounds or method == "HYP":
            break
        rounds += 1
        worst = bad[np.argsort(-viol[bad])][:30]
        x = np.concatenate([x, xd[worst]]); y = np.concatenate([y, yd[worst]]); o = np.argsort(x); x, y = x[o], y[o]
    if method == "HYP":  # C = max over all staircase points => dominance by construction
        terms = [(float(np.max(xd * yd)), -1.0)]; g = env_eval(terms, xd); bad = []
    xmin, xmax = float(xd.min()), float(xd.max())
    env = Envelope(kind, terms, xmin, xmax)
    gd = env(xd)
    info.update({"rounds": rounds, "n_fit": len(x), "n_dom": len(xd), "dominates_all": bool(np.all(gd >= yd - 1e-12)),
                 "sse_fit": float(np.sum((env(x) - y) ** 2)), "maxerr_fit": float(np.max(env(x) - y)),
                 "rsse_dom": float(np.sum(((env(xd) - yd) / yd) ** 2)), "maxrel_dom": float(np.max((env(xd) - yd) / yd)),
                 "sse_dom": float(np.sum((gd - yd) ** 2)), "maxerr_dom": float(np.max(gd - yd)),
                 "logarea": float(np.mean(np.log10(gd / yd))), "total_time": time.time() - t0})
    return env, info

def dominance_on(env, gen, imp, mode="corner", region=None):
    """Test-set check: fraction of staircase points (within the envelope's range) violated."""
    _, far, frr = roc(gen, imp)
    x, y = staircase_points(far, frr, mode)
    m = (x >= env.xmin) & (x <= env.xmax)
    if region is not None: m &= (x >= region[0]) & (x <= region[1])
    x, y = x[m], y[m]
    if len(x) == 0: return 0.0, 0.0
    v = y - env(x)
    return float(np.mean(v > 1e-12)), float(max(0.0, v.max()))

def design_and_evaluate(envs, alpha, tr, te, order):
    """tr/te: dicts matcher -> (scores 2-D, genuine mask). Returns metrics dict."""
    d = gp_design([envs[m] for m in order], alpha)
    if d is None: return {"feasible": False}
    imp_tr = [np.sort(tr[m][0][~tr["_G"]]) for m in order]; gen_tr = [np.sort(tr[m][0][tr["_G"]]) for m in order]
    thr = recover_thresholds(d, imp_tr)
    emp = empirical_stage_rates(thr, gen_tr, imp_tr)
    out = {"feasible": True, "frr_pred_gp": d["frr_pred"], "far_model_gp": d["far_model"],
           "frr_pred_exact": composition(d["stages"], "r_acc", "r_rej", exact=True),
           "frr_emp_model": composition(emp, "r_acc", "r_rej", exact=True),
           "far_emp_model": composition(emp, "a_acc", "a_rej", exact=True)}
    fa, fr, sg, si = simulate([tr[m][0] for m in order], tr["_G"], thr)
    out.update(far_train=fa, frr_train=fr)
    fa, fr, sg, si = simulate([te[m][0] for m in order], te["_G"], thr)
    out.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si)
    return out

def evaluate_rule(name, alpha, tr, te, order):
    sc_tr = [tr[m][0] for m in order]; G = tr["_G"]
    if name == "S1-Marcialis":
        if len(order) < 2: return {"feasible": False}
        thr = rule_marcialis(sc_tr, G, alpha)
    elif name == "S2-Symmetric":
        if len(order) < 2: return {"feasible": False}
        thr = rule_symmetric(sc_tr, G, alpha)
    elif name == "S4-Direct":
        if len(order) != 2: return {"feasible": False}
        xmin = 1.0 / (~G).sum()
        thr = rule_direct(sc_tr, G, alpha, xmin)
    else:
        raise ValueError(name)
    if thr is None: return {"feasible": False}
    fa, fr, _, _ = simulate(sc_tr, G, thr)
    out = {"feasible": True, "far_train": fa, "frr_train": fr}
    fa, fr, sg, si = simulate([te[m][0] for m in order], te["_G"], thr)
    out.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si)
    return out

def evaluate_parallel(alpha, tr, te, matchers):
    t, z_tr, z_te = parallel_sum([tr[m][0] for m in matchers], tr["_G"], [te[m][0] for m in matchers], alpha)
    G, Gt = tr["_G"], te["_G"]
    return {"feasible": True, "far_train": float((z_tr[~G] >= t).mean()), "frr_train": float((z_tr[G] < t).mean()),
            "far_test": float((z_te[~Gt] >= t).mean()), "frr_test": float((z_te[Gt] < t).mean()),
            "stages_gen": float(len(matchers)), "stages_imp": float(len(matchers))}

def make_blocks(D, matchers, seed, train_frac=0.5):
    r_tr, c_tr, r_te, c_te = split_subjects(D["row_subject"], D["col_subject"], seed, train_frac)
    tr = {"_G": D["genuine"][np.ix_(r_tr, c_tr)]}; te = {"_G": D["genuine"][np.ix_(r_te, c_te)]}
    for m in matchers:
        S = D["S_" + m]
        tr[m] = (S[np.ix_(r_tr, c_tr)],); te[m] = (S[np.ix_(r_te, c_te)],)
    return tr, te

def all_orders(matchers):
    out = []
    for k in range(1, len(matchers) + 1):
        out += list(itertools.permutations(matchers, k))
    return out
