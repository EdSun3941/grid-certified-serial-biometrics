"""Revision (reviewer 1): acceptance rate of a zero-knowledge impostor who presents a PERFECT spoof of one
modality (that matcher's score is drawn from the genuine test scores, all other scores stay impostor scores).
Evaluated on the test half for the designs selected in E3b (ties averaged), alpha = 1e-3.
Output: results/tables/T_rev_spoof.csv"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from experiment_core import make_blocks
from serial_design import simulate, llr_function
from data import MATCHERS, data_file

METHODS = ["LR-P1-N2", "S1-Marcialis", "S2-Symmetric", "S3-SPRT", "S4-Direct", "P0-Parallel", "P1-LLR", "P2-LogReg"]
ALPHA = 1e-3
sel = pd.read_csv("../results/tables/T_rev_selected.csv")
rows = []
for ds in ["fing_x_face", "fing_x_fing", "face_x_face"]:
    M = MATCHERS[ds]; D = dict(np.load(data_file(ds)))
    for seed in range(10):
        e3 = pd.read_csv(f"../results/E3b/main_{ds}_s{seed}.csv"); e3 = e3[(e3.feasible == True) & np.isclose(e3.alpha, ALPHA)]
        tr, te = make_blocks(D, M, seed); G, Gt = tr["_G"], te["_G"]; I = ~Gt
        rng = np.random.default_rng(7 + seed)
        spoofed = {m: rng.choice(te[m][0][Gt], size=Gt.shape, replace=True) for m in M}   # genuine-distributed scores
        llr = {m: llr_function(tr[m][0][G], tr[m][0][~G]) for m in M}
        for meth in METHODS:
            s = sel[(sel.dataset == ds) & (sel.seed == seed) & np.isclose(sel.alpha, ALPHA) & (sel.method == meth)]
            if len(s) == 0: continue
            orders = s.orders.iloc[0].split("|")
            rates = {m: [] for m in M}
            for order in orders:
                r = e3[(e3.method == meth) & (e3.order == order)].iloc[0]; thr = json.loads(r.thr)
                if meth.startswith("P"):                          # parallel fusion: rebuild the fused score
                    for m_sp in M:
                        sc = {m: (spoofed[m] if m == m_sp else te[m][0]) for m in M}
                        if meth == "P0-Parallel":
                            f = sum((sc[m] - tr[m][0][~G].mean()) / (tr[m][0][~G].std() + 1e-12) for m in M)
                        elif meth == "P1-LLR":
                            f = sum(llr[m](sc[m]) for m in M)
                        else:
                            X_tr = np.stack([tr[m][0].ravel() for m in M], 1); mu_x, sd_x = X_tr.mean(0), X_tr.std(0) + 1e-12
                            y = G.ravel(); rg = np.random.default_rng(seed); imp_idx = np.nonzero(~y)[0]
                            sub = np.concatenate([np.nonzero(y)[0], rg.choice(imp_idx, min(len(imp_idx), 200000), replace=False)])
                            lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit((X_tr[sub] - mu_x) / sd_x, y[sub])
                            X = np.stack([sc[m].ravel() for m in M], 1); f = lr.decision_function((X - mu_x) / sd_x).reshape(Gt.shape)
                        rates[m_sp].append(float((f[I] >= thr[0]).mean()))
                    continue
                chain = order.split(">"); thr = [tuple(t) if isinstance(t, list) else t for t in thr]
                for m_sp in M:
                    if m_sp not in chain: rates[m_sp].append(np.nan); continue
                    sc = [spoofed[m] if m == m_sp else te[m][0] for m in chain]
                    if meth == "S3-SPRT":
                        c = 0; cum = []
                        for m, x in zip(chain, sc): c = c + llr[m](x); cum.append(c)
                        sc = cum
                    fa, _, _, _ = simulate(sc, Gt, thr); rates[m_sp].append(fa)
            per_mod = {m: np.nanmean(v) if len(v) and not np.all(np.isnan(v)) else np.nan for m, v in rates.items()}
            vals = [v for v in per_mod.values() if np.isfinite(v)]
            rows.append({"dataset": ds, "seed": seed, "alpha": ALPHA, "method": meth, "spoof_max": max(vals), "spoof_mean": np.mean(vals),
                         **{f"spoof_{m}": v for m, v in per_mod.items()}})
        print(ds, seed, flush=True)
    del D
t = pd.DataFrame(rows)
agg = t.groupby(["dataset", "method"]).agg(spoof_max=("spoof_max", "mean"), spoof_mean=("spoof_mean", "mean"), n=("seed", "size")).reset_index()
agg["source"] = "results/E3b/main_*.csv thresholds + test scores; spoof_rev.py"
t.to_csv("../results/tables/T_rev_spoof_raw.csv", index=False); agg.to_csv("../results/tables/T_rev_spoof.csv", index=False)
print(agg.round(4).to_string())
