"""Revision check (Section VI-B claims): on the 40 P1 fitting sets of the solver comparison (E2, N = 2; optimum from the independent reference),
(a) value of the continuous-exponent monomial (grid spacing 1e-4) relative to the grid optimum with N = 2, and
(b) the exponents carrying weight in the optimal solution of the convexified problem (eq. conv) at the root.
Output: results/tables/T_rev_monoc.csv"""
import glob, numpy as np, pandas as pd
from experiment_core import fitting_data, make_blocks, B_DEFAULT
from envelope_solvers import fit_monomial, _design, _convexified_p1, objective_any
from data import MATCHERS, data_file
v2 = pd.concat([pd.read_csv(f) for f in glob.glob("../results/E2ref/ref_*.csv")], ignore_index=True)   # independent exact optimum
v2 = v2[v2.obj == "P1"]
Bc = np.round(np.arange(-3, 1e-9, 1e-4), 4); B = np.asarray(B_DEFAULT); rows = []
for ds in ["fing_x_face", "fing_x_fing", "face_x_face"]:
    D = dict(np.load(data_file(ds))); M = MATCHERS[ds]
    for seed in sorted(v2[v2.dataset == ds].seed.unique()):
        tr, _ = make_blocks(D, M, int(seed)); G = tr["_G"]
        for m in M:
            S = tr[m][0]; x, y, _, _ = fitting_data(S[G], S[~G])
            opt = v2[(v2.dataset == ds) & (v2.seed == seed) & (v2.matcher == m)].value.iloc[0]
            _, vc = fit_monomial(x, y, Bc, "P1"); _, vg = fit_monomial(x, y, B, "P1")
            Phi, yv = _design(x, y, B, False); cols = np.arange(len(B))
            val, mu, w = _convexified_p1(Phi, yv, [cols, cols], 2)
            cw = w * Phi.max(axis=0); cw = cw / cw.max(); sup = B[cw > 1e-3]      # contribution of each exponent (weight x column scale)
            rows.append(dict(dataset=ds, seed=int(seed), matcher=m, opt_N2=opt, mono_grid=vg, mono_cont=vc,
                             rel_cont=(vc - opt) / opt, rel_grid=(vg - opt) / opt, conv_value=val,
                             support=" ".join(f"{b:.2f}" for b in sup), n_support=len(sup)))
            print(rows[-1], flush=True)
    del D
t = pd.DataFrame(rows); t["source"] = "check_monoc_weights.py on the E2 fitting sets; results/E2ref"
t.to_csv("../results/tables/T_rev_monoc.csv", index=False)
print(t[["rel_cont", "rel_grid", "n_support"]].describe())
