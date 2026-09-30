"""IJIS revision (reviewer 3, comment 4): comparison of the methods at MATCHED realized FAR.
For every selected design of E3b (tie-averaged orders of T_rev_selected.csv), the earlier stage thresholds are kept and
the final threshold is re-set on the TEST half to the most permissive value whose test FAR does not exceed alpha
(an oracle operating point, identical in construction for every method; parallel fusion has only this threshold).
The test FRR at this matched FAR measures the discriminative quality of each design at equal FAR, not deployable
performance.  Paired differences to the proposed design: corrected resampled t-test, Holm within each setting.
Output: results/tables/T_rev_matchedfar.csv (per split), T_rev_matchedfar_system.csv, T_rev_matchedfar_stats.csv
IJIS v02: `python matched_far.py --fresh --part cloud|local` does the same for the designs of results/E3fresh with the
paper calibration (calib boot; new splits 10-29 of D1-D3 and splits 0-9 of D4), each split on the machine that computed it,
and `python matched_far.py --fresh --part combine` merges the rows -> results/tables/T_fresh_matchedfar*.csv
(`--part all` evaluates every split on this machine, for a reproduction on a single machine)"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from experiment_core import make_blocks
from serial_design import llr_function, simulate
from data import MATCHERS, data_file
from analyze_rev import corrected_t, holm, MAIN, PRIMARY

METHODS = [MAIN] + PRIMARY
MISMATCH = []


def matched(scores, G, pre, alpha):
    """Final threshold on the test half: most permissive t with test FAR <= alpha (earlier stages fixed)."""
    S = len(scores); decided = np.zeros(G.shape, bool); acc = np.zeros(G.shape, bool)
    for s in range(S - 1):
        ta, tr = pre[s]; und = ~decided
        a = und & (scores[s] >= ta); r = und & (scores[s] < tr); acc |= a; decided |= a | r
    I = ~G; n_imp = int(I.sum()); n_gen = int(G.sum())
    a0 = int((acc & I).sum()); k = int(np.floor(alpha * n_imp + 1e-9)) - a0
    if k < 0: return np.nan, np.nan, False
    si = np.sort(scores[-1][(~decided) & I])[::-1]
    t = -np.inf if k >= len(si) else float(np.nextafter(si[k], np.inf))
    fin = (~decided) & (scores[-1] >= t)
    accept = acc | fin
    return float((~accept[G]).mean()), float(accept[I].mean()), True


def main(fresh=False, part=None):
    if fresh:
        # Each split is evaluated on the machine that computed its designs (results/E3fresh_provenance.csv): the SPRT
        # thresholds are cumulative-LLR values, and last-bit differences between machines flip exact ties of the integer
        # fingerprint scores.  part = "cloud" or "local" writes the per-split rows; part = "combine" merges and tests.
        sel = pd.read_csv("../results/tables/T_fresh_selected.csv"); sel = sel[sel.method.isin(METHODS) & (sel.calib == "boot")]
        if part == "all":                                           # single-machine reproduction: every split here
            mine = {f"{d}_s{s_}" for d, r in [("lfw_x_fing", range(10))] + [(d, range(10, 30)) for d in ("fing_x_face", "fing_x_fing", "face_x_face")] for s_ in r}
        else:
            prov = pd.read_csv("../results/E3fresh_provenance.csv"); prov["key"] = prov.file.str.replace("fresh_", "").str.replace(".csv", "")
            mine = set(prov[prov.machine.str.startswith(part)].key) if part in ("cloud", "local") else set()
        SETS = [(d, [s_ for s_ in seeds if f"{d}_s{s_}" in mine]) for d, seeds in
                [("lfw_x_fing", range(10)), ("fing_x_face", range(10, 30)), ("fing_x_fing", range(10, 30)), ("face_x_face", range(10, 30))]]
        PFX, SRC = "T_fresh_matchedfar", "E3fresh thresholds (calib boot, earlier stages) + test scores; matched_far.py --fresh"
        if part == "combine":
            t = pd.concat([pd.read_csv(f"../results/tables/{PFX}_rows_{m}.csv") for m in ("cloud", "local")], ignore_index=True)
            return summarize(t, PFX, SRC)
    else:
        sel = pd.read_csv("../results/tables/T_rev_selected.csv"); sel = sel[sel.method.isin(METHODS)]
        SETS = [(d, range(10)) for d in ["fing_x_face", "fing_x_fing", "face_x_face"]]
        PFX, SRC = "T_rev_matchedfar", "E3b thresholds (earlier stages) + test scores; matched_far.py"
    rows = []
    for ds, seeds in SETS:
        if len(seeds) == 0: continue
        M = MATCHERS[ds]; D = dict(np.load(data_file(ds)))
        for seed in seeds:
            if fresh:
                e3 = pd.read_csv(f"../results/E3fresh/fresh_{ds}_s{seed}.csv"); e3 = e3[(e3.feasible == True) & (e3.calib == "boot")]
            else:
                e3 = pd.read_csv(f"../results/E3b/main_{ds}_s{seed}.csv"); e3 = e3[e3.feasible == True]
            tr, te = make_blocks(D, M, seed); G, Gt = tr["_G"], te["_G"]
            llr = {m: llr_function(tr[m][0][G], tr[m][0][~G]) for m in M}
            z = sum((te[m][0] - tr[m][0][~G].mean()) / (tr[m][0][~G].std() + 1e-12) for m in M)
            l = sum(llr[m](te[m][0]) for m in M)
            X_tr = np.stack([tr[m][0].ravel() for m in M], 1); mu_x, sd_x = X_tr.mean(0), X_tr.std(0) + 1e-12; y = G.ravel()
            rng = np.random.default_rng(seed); imp_idx = np.nonzero(~y)[0]           # identical to run_cal2.py
            sub = np.concatenate([np.nonzero(y)[0], rng.choice(imp_idx, min(len(imp_idx), 200000), replace=False)])
            lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit((X_tr[sub] - mu_x) / sd_x, y[sub])
            X_te = np.stack([te[m][0].ravel() for m in M], 1); f_lr = lr.decision_function((X_te - mu_x) / sd_x).reshape(Gt.shape)
            del X_tr, X_te
            fused = {"P0-Parallel": z, "P1-LLR": l, "P2-LogReg": f_lr}
            for _, s in sel[(sel.dataset == ds) & (sel.seed == seed)].iterrows():
                vals = []
                for order in s.orders.split("|"):
                    r = e3[(e3.method == s.method) & (e3.order == order) & np.isclose(e3.alpha, s.alpha)]
                    if len(r) == 0: continue
                    thr = json.loads(r.iloc[0].thr)
                    if s.method in fused:
                        f_ = fused[s.method]; fa_ = float((f_[~Gt] >= thr[0]).mean()); fr_ = float((f_[Gt] < thr[0]).mean())
                        if abs(fa_ - r.iloc[0].far_test) > 1e-12 or abs(fr_ - r.iloc[0].frr_test) > 1e-12: MISMATCH.append((ds, seed, s.method, order))
                        vals.append(matched([f_], Gt, [], s.alpha)); continue
                    chain = order.split(">"); sc = [te[m][0] for m in chain]
                    if s.method == "S3-SPRT":
                        c = 0; cum = []
                        for m, x in zip(chain, sc): c = c + llr[m](x); cum.append(c)
                        sc = cum
                    pre = [tuple(t) for t in thr[:-1]]
                    fa_, fr_, _, _ = simulate(sc, Gt, pre + [thr[-1]])
                    if abs(fa_ - r.iloc[0].far_test) > 1e-12 or abs(fr_ - r.iloc[0].frr_test) > 1e-12: MISMATCH.append((ds, seed, s.method, order))
                    vals.append(matched(sc, Gt, pre, s.alpha))
                ok = [v for v in vals if v[2]]
                rows.append(dict(dataset=ds, seed=seed, alpha=s.alpha, method=s.method, n_orders=len(vals), n_feasible=len(ok),
                                 frr_matched=np.mean([v[0] for v in ok]) if ok else np.nan,
                                 far_matched=np.mean([v[1] for v in ok]) if ok else np.nan,
                                 frr_deployed=s.frr_test, far_deployed=s.far_test))
            print(ds, seed, flush=True)
        del D
    print("reconstruction mismatches with the stored results:", len(MISMATCH), MISMATCH[:5])
    assert not MISMATCH, "test scores do not reproduce the stored results"
    t = pd.DataFrame(rows); t["source"] = SRC
    if fresh and part != "all":
        t["machine"] = part; t.to_csv(f"../results/tables/{PFX}_rows_{part}.csv", index=False); return
    summarize(t, PFX, SRC)


def summarize(t, PFX, SRC):
    t.to_csv(f"../results/tables/{PFX}.csv", index=False)
    agg = t.groupby(["dataset", "alpha", "method"]).agg(n=("seed", "size"), n_feasible_all=("n_feasible", lambda v: int((v > 0).sum())),
                                                        frr_matched=("frr_matched", "mean"), far_matched=("far_matched", "mean"),
                                                        frr_deployed=("frr_deployed", "mean"), far_deployed=("far_deployed", "mean")).reset_index()
    agg["far_matched_over_alpha"] = agg.far_matched / agg.alpha; agg["source"] = f"{PFX}.csv"
    agg.to_csv(f"../results/tables/{PFX}_system.csv", index=False)
    st = []
    for (ds, a), g in t.groupby(["dataset", "alpha"]):
        mv = g[g.method == MAIN].set_index("seed").frr_matched.dropna()
        for m in PRIMARY:
            o = g[g.method == m].set_index("seed").frr_matched.dropna(); c = mv.index.intersection(o.index)
            if len(c) < 5: st.append(dict(dataset=ds, alpha=a, method=m, n=len(c))); continue
            d = (mv[c] - o[c]).values; mean, ci, p = corrected_t(d)
            st.append(dict(dataset=ds, alpha=a, method=m, n=len(c), mean_diff=mean, ci_lo=ci[0], ci_hi=ci[1],
                           rel_diff=mean / o[c].mean() if o[c].mean() > 0 else np.nan, p_corr_t=p,
                           n_better=int((d < 0).sum()), n_worse=int((d > 0).sum())))
    st = pd.DataFrame(st); st["p_holm"] = np.nan
    for _, g in st[st.p_corr_t.notna()].groupby(["dataset", "alpha"]):
        st.loc[g.index, "p_holm"] = holm(g.p_corr_t.values)
    st["source"] = f"{PFX}.csv; corrected resampled t-test, Holm within setting"
    st.to_csv(f"../results/tables/{PFX}_stats.csv", index=False)
    print(agg.pivot_table(index=["dataset", "alpha"], columns="method", values="frr_matched").round(4).to_string())
    print(st[["dataset", "alpha", "method", "n", "mean_diff", "ci_lo", "ci_hi", "p_holm", "n_better", "n_worse"]].round(4).to_string())


if __name__ == "__main__":
    import sys
    part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else None
    main(fresh="--fresh" in sys.argv, part=part)
