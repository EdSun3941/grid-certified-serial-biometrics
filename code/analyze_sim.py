"""IJIS v08: summary of the controlled simulation (results/E9sim/sim_*.csv, written by sim_controlled.py).
Per training size N, genuine correlation rho_g and alpha: coverage of the population FAR requirement by the deployed
(selected, tie-averaged) proposed design and by all calibrated designs, Wilson 95% interval of the selected-design
coverage, mean population FAR / alpha, how often the step-(iii) prediction bounds the population FRR and the joint
training FRR, and how often it bounds the training FRR under the product of the per-matcher training distributions
(Lemma, Proposition 1).  v12: the coverage of a replicate is tie-averaged and can be fractional; the Wilson interval
treats it as a Bernoulli outcome, which is conservative because an outcome in [0, 1] with mean p has variance at most
p(1 - p); the number of replicates with a fractional outcome is reported (frac_reps, xfit_frac_reps).
Output: results/tables/T_sim.csv"""
import glob, numpy as np, pandas as pd
R = "../results"; OUT = f"{R}/tables"


def wilson(k, n, z=1.959964):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    fs = sorted(glob.glob(f"{R}/E9sim/sim_*.csv")); df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    rows = []
    for (N, rho, a), g0 in df.groupby(["N", "rho_g", "alpha"]):
        g = g0[g0.calib == "boot"]; x = g0[g0.calib == "xfit"]
        reps = g.rep.nunique(); f = g[g.feasible == True]
        sel = f[f.selected == True].copy()
        w = 1.0 / sel.groupby("rep").order.transform("size")                     # tie-averaging weights
        ok = (sel.far_pop <= a).astype(float)
        k_sel = float((w * ok).sum()); n_sel = float(w.sum())
        y_r = (w * ok).groupby(sel.rep).sum(); frac = int(((y_r > 1e-12) & (y_r < 1 - 1e-12)).sum())
        lo, hi = wilson(k_sel, n_sel)
        two = f[f.n_stages == 2]
        row = dict(N=N, rho_g=rho, rho_i=g.rho_i.iloc[0], alpha=a, reps=reps, reps_deployed=int(sel.rep.nunique()), frac_reps=frac,
                   cov_selected=k_sel / n_sel, cov_lo=lo, cov_hi=hi,
                   far_pop_over_alpha=float((w * sel.far_pop).sum() / n_sel / a),
                   cov_all=float((f.far_pop <= a).mean()), n_designs=len(f),
                   train_ok_selected=float((w * (sel.far_train <= a)).sum() / n_sel),
                   pred_ge_pop_selected=float((w * (sel.frr_pred >= sel.frr_pop)).sum() / n_sel),
                   pred_ge_pop_all=float((f.frr_pred >= f.frr_pop).mean()),
                   pred_ge_pop_two_stage=float((two.frr_pred >= two.frr_pop).mean()),
                   pred_ge_joint_all=float((f.frr_pred >= f.frr_train - 1e-12).mean()),
                   pred_ge_joint_two_stage=float((two.frr_pred >= two.frr_train - 1e-12).mean()),
                   lemma_all=float((f.frr_pred >= f.frr_prod - 1e-12).mean()), lemma_n=len(f),
                   frr_pop_selected=float((w * sel.frr_pop).sum() / n_sel),
                   pred_over_pop_selected=float(np.median(sel.frr_pred / sel.frr_pop)),
                   two_stage_selected=float((w * (sel.n_stages == 2)).sum() / n_sel))
        if len(x):                                                                # held-out arm (order fixed on fold A)
            xs = x[x.feasible == True].copy(); wx = 1.0 / xs.groupby("rep").order.transform("size")
            dep = xs.deployed.astype(bool); n_dep = float((wx * dep).sum())
            okx = ((xs.far_pop <= a) & dep).astype(float); k_x = float((wx * okx).sum())
            lo2, hi2 = wilson(k_x, n_dep)
            yd = (wx * dep).groupby(xs.rep).sum(); xfrac = int(((yd > 1e-12) & (yd < 1 - 1e-12)).sum())
            row.update(xfit_frac_reps=xfrac, xfit_reps=int(x.rep.nunique()), xfit_deployed=n_dep / x.rep.nunique(), xfit_cov_deployed=k_x / n_dep if n_dep else np.nan,
                       xfit_cov_lo=lo2, xfit_cov_hi=hi2, xfit_cov_all=k_x / x.rep.nunique(),
                       xfit_far_pop_over_alpha=float((wx[dep] * xs.far_pop[dep]).sum() / n_dep / a) if n_dep else np.nan,
                       xfit_frr_pop=float((wx[dep] * xs.frr_pop[dep]).sum() / n_dep) if n_dep else np.nan)
        rows.append(row)
    T = pd.DataFrame(rows); T["source"] = f"results/E9sim/sim_*.csv ({len(fs)} files)"; T.to_csv(f"{OUT}/T_sim.csv", index=False)
    pd.set_option("display.width", 250); print(T.drop(columns="source").round(3).to_string())


if __name__ == "__main__":
    main()
