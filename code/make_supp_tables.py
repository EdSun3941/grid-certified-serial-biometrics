"""Generate the LaTeX tables of the supplementary material directly from results/tables/*.csv
(no manual transcription).  Output: ../paper/supp_tables.tex"""
import os
import numpy as np, pandas as pd

T = "../results/tables"; DS = {"fing_x_face": "D1", "fing_x_fing": "D2", "face_x_face": "D3"}
ALPHA = {1e-2: "$10^{-2}$", 1e-3: "$10^{-3}$", 1e-4: "$10^{-4}$"}
out = []

def f4(v): return "--" if pd.isna(v) else f"{v:.4f}"

# S1: uncalibrated (TDSC-style) system results
t = pd.read_csv(f"{T}/T_system_plain.csv")
cols = [("LR-P1-N2", "Proposed"), ("HYP", "HYP"), ("MONO", "MONO"), ("S1-Marcialis", "Marcialis"), ("S2-Symmetric", "Symmetric"), ("S4-Direct", "Direct"), ("P0-Parallel", "Parallel")]
out.append(r"\begin{table*}[!t]\centering\caption{Uncalibrated Designs (Thresholds at the Point Estimate of the Training FAR, as in the Prior GP Design): Test FRR (Mean Over Ten Splits) and, in Parentheses, Share of Splits With Test FAR $\le\alpha$}\label{tab:s-plain}\footnotesize\setlength{\tabcolsep}{3pt}")
out.append(r"\begin{tabular}{@{}ll" + "c" * len(cols) + r"@{}}\toprule Set & $\alpha$ & " + " & ".join(c for _, c in cols) + r" \\\midrule")
for ds in DS:
    for a in sorted(t[t.dataset == ds].alpha.unique(), reverse=True):
        cells = []
        for m, _ in cols:
            r = t[(t.dataset == ds) & np.isclose(t.alpha, a) & (t.method == m)]
            if len(r) == 0 or pd.isna(r.frr_test.iloc[0]): cells.append("--"); continue
            r = r.iloc[0]; cells.append(f"{r.frr_test:.4f} ({100 * r.far_ok_test_sel:.0f}\\%)")
        out.append(f"{DS[ds]} & {ALPHA[a]} & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
out[-1] = r"\bottomrule\end{tabular}\end{table*}"

# S3: fitting metrics per family
f = pd.read_csv(f"{T}/T_fit.csv")
fam = [("HYP", "HYP"), ("MONO", "MONO"), ("MONO-rel", "MONO-rel"), ("NLS-N2", "NLS"), ("DE-N2", "DE"), ("MAXMONO-K3", "MAXMONO"),
       ("LR-P1-N2", "LR-BB P1"), ("LR-P1-N2-rel", "LR-BB P1-rel"), ("LR-P2-N2", "LR-BB P2"), ("LR-P2-N2-rel", "LR-BB P2-rel"), ("LR-P1-N3", "LR-BB P1, $N=3$")]
out.append(r"\begin{table*}[!t]\centering\caption{Envelope Fitting on the Training Halves: SSE Ratio to LR-BB P1 (Geometric Mean), Mean $\log_{10}$ Overestimation Over All Corners, Share of Test Corners Above the Envelope, and Fitting Time (s, Including Constraint Generation)}\label{tab:s-fit}\footnotesize\setlength{\tabcolsep}{3pt}")
out.append(r"\begin{tabular}{@{}l" + "cccc" * 3 + r"@{}}\toprule & \multicolumn{4}{c}{D1} & \multicolumn{4}{c}{D2} & \multicolumn{4}{c}{D3}\\\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
out.append(r"Envelope" + r" & SSE ratio & log-over & test viol. & time" * 3 + r" \\\midrule")
def g(v):
    return f"{v:.2e}".replace("e+0", "e").replace("e+", "e") if v >= 1e3 else f"{v:.2f}"
for m, lab in fam:
    cells = []
    for ds in DS:
        r = f[(f.dataset == ds) & (f.method == m)].iloc[0]
        cells += [g(r.sse_ratio_gm), f"{r.logarea:.2f}", f"{100 * r.test_viol:.1f}\\%", f"{r.fit_time:.2f}"]
    out.append(lab + " & " + " & ".join(cells) + r" \\")
out.append(r"\bottomrule\end{tabular}\end{table*}")

# S4: scalability
sc = pd.read_csv(f"{T}/T_scalability.csv")
out.append(r"\begin{table*}[!t]\centering\caption{Scalability (Seed 0, P1, Single Core, 600-s Limit per Solver). Enumeration Times Marked $^{e}$ Are Extrapolated From the Enumerated Fraction}\label{tab:s-scal}\footnotesize\setlength{\tabcolsep}{3pt}")
out.append(r"\begin{tabular}{@{}llrrrrrrrr@{}}\toprule Matcher & Varied & $N$ & $|\mathcal{D}|$ & $m$ & LR-BB (s) & Gap (\%) & Enumeration (s) & Speed-up & Grid vs.\ DE (\%) \\\midrule")
lab = {"fing_x_face": "D1 face C", "fing_x_fing": "D2 left index"}
for (ds, w), gdf in sc.groupby(["dataset", "which"], sort=False):
    for _, r in gdf.sort_values("setting").iterrows():
        en = f"{r.enum_time:.1f}" + ("" if r.enum_complete else "$^{e}$")
        out.append(f"{lab[ds]} & {w} & {int(r.N)} & {int(r.nD)} & {int(r.nB)} & {r.lr_time:.2f} & {100 * r.lr_cert_gap:.1f} & {en} & "
                   f"{'--' if pd.isna(r.speedup) else f'{r.speedup:.1f}'} & {'--' if pd.isna(r.grid_vs_de) else f'{100 * r.grid_vs_de:.2f}'} \\\\")
    out.append(r"\midrule")
out[-1] = r"\bottomrule\end{tabular}\end{table*}"

# S5: reject-threshold location
rf = pd.read_csv(f"{T}/T_reject_far.csv")
out.append(r"\begin{table}[!t]\centering\caption{Stage FAR at the Accept and Reject Thresholds of the Uncalibrated Proposed GP Designs (All Multi-Stage Orders, Ten Splits)}\label{tab:s-reject}\footnotesize")
out.append(r"\begin{tabular}{@{}llrccc@{}}\toprule Set & $\alpha$ & Stages & \makecell{Median FAR\\at accept} & \makecell{Median FAR\\at reject} & \makecell{Reject FAR\\$>10^{-1}$} \\\midrule")
for ds in DS:
    for _, r in rf[rf.dataset == ds].sort_values("alpha", ascending=False).iterrows():
        out.append(f"{DS[ds]} & {ALPHA[r.alpha]} & {int(r.n)} & {r.a_acc_median:.2e} & {r.a_rej_median:.3f} & {100 * r.frac_a_rej_gt_0p1:.1f}\\% \\\\")
out.append(r"\bottomrule\end{tabular}\end{table}")

# S6: noise robustness (D1, uncalibrated)
nz = pd.read_csv(f"{T}/T_noise.csv")
out.append(r"\begin{table}[!t]\centering\caption{Robustness of D1 Designs to Additive Gaussian Test-Score Noise (SD = 5\% or 10\% of the Impostor-Score SD; Uncalibrated): Test FAR / FRR}\label{tab:s-noise}\footnotesize\setlength{\tabcolsep}{3pt}")
out.append(r"\begin{tabular}{@{}llccc@{}}\toprule $\alpha$ & Envelope & No noise & 5\% & 10\% \\\midrule")
for a in [1e-2, 1e-3]:
    for m, lab2 in [("LR-P1-N2", "LR-BB P1"), ("MONO", "MONO"), ("DE-N2", "DE"), ("HYP", "HYP")]:
        r = nz[np.isclose(nz.alpha, a) & (nz.method == m)]
        if len(r) == 0: continue
        r = r.iloc[0]
        out.append(f"{ALPHA[a]} & {lab2} & {r.far_test:.4f} / {r.frr_test:.4f} & {r.far_test_n05:.4f} / {r.frr_test_n05:.4f} & {r.far_test_n10:.4f} / {r.frr_test_n10:.4f} \\\\")
out.append(r"\bottomrule\end{tabular}\end{table}")


# S7: Clopper-Pearson calibrated results (first version of the paper; order selected by predicted FRR)
cp = pd.read_csv(f"{T}/T_system_cal.csv")
cols2 = [("LR-P1-N2", "Proposed"), ("HYP", "HYP"), ("S1-Marcialis", "Marcialis"), ("S2-Symmetric", "Symmetric"), ("S4-Direct", "Direct"), ("P0-Parallel", "Sum")]
out.append(r"\begin{table}[!t]\centering\caption{Designs With Pair-Level Clopper--Pearson Calibration (GP Designs Selected by Predicted FRR, Rules by Training FRR, First Tie Kept): Test FRR}\label{tab:s-cp}\footnotesize\setlength{\tabcolsep}{2.5pt}")
out.append(r"\begin{tabular}{@{}ll" + "c" * len(cols2) + r"@{}}\toprule Set & $\alpha$ & " + " & ".join(c for _, c in cols2) + r" \\\midrule")
for ds in DS:
    for a in sorted(cp[cp.dataset == ds].alpha.unique(), reverse=True):
        cells = []
        for m, _ in cols2:
            r = cp[(cp.dataset == ds) & np.isclose(cp.alpha, a) & (cp.method == m)]
            cells.append("--" if len(r) == 0 or pd.isna(r.frr_test.iloc[0]) else f"{r.frr_test.iloc[0]:.4f}")
        out.append(f"{DS[ds]} & {ALPHA[a]} & " + " & ".join(cells) + r" \\")
out.append(r"\bottomrule\end{tabular}\end{table}")

# S8: compliance by chain type (subject bootstrap)
cv = pd.read_csv(f"{T}/T_rev_conservativeness.csv")
out.append(r"\begin{table}[!t]\centering\caption{Subject-Bootstrap Calibration: Share of Proposed Designs With Test FAR $\le\alpha$, Median Test FAR$/\alpha$, and Share With Test FRR $\le$ Prediction, by Chain Type}\label{tab:s-cons}\footnotesize\setlength{\tabcolsep}{3pt}")
out.append(r"\begin{tabular}{@{}llrrccc@{}}\toprule Set & $\alpha$ & Chains & $n$ & FAR ok & FAR$/\alpha$ & FRR ok \\\midrule")
for ds in DS:
    for a in sorted(cv[cv.dataset == ds].alpha.unique(), reverse=True):
        for _, r in cv[(cv.dataset == ds) & np.isclose(cv.alpha, a)].iterrows():
            out.append(f"{DS[ds]} & {ALPHA[a]} & {r.chain} & {int(r.n)} & {100 * r.far_ok_test:.0f}\\% & {r.far_ratio_median:.2f} & {100 * r.frr_ok_test:.0f}\\% \\\\")
out.append(r"\bottomrule\end{tabular}\end{table}")

# S9: decomposition with IQR
dc = pd.read_csv(f"{T}/T_rev_decomposition.csv")
out.append(r"\begin{table*}[!t]\centering\caption{Per-Design Decomposition of the Predicted System FRR (Uncalibrated Proposed Designs, All Orders): Median [Interquartile Range] of Each Successive Difference}\label{tab:s-decomp}\footnotesize\setlength{\tabcolsep}{3pt}")
out.append(r"\begin{tabular}{@{}llrcccccc@{}}\toprule Set & $\alpha$ & $n$ & GP objective & Series bound & Envelope & Dependence & Generalization & Test FRR \\\midrule")
for ds in DS:
    for _, r in dc[dc.dataset == ds].sort_values("alpha", ascending=False).iterrows():
        cells = [f"{r[k + '_median']:+.3f} [{r[k + '_q25']:+.3f}, {r[k + '_q75']:+.3f}]" for k in ["series bound", "envelope", "dependence", "generalization"]]
        out.append(f"{DS[ds]} & {ALPHA[r.alpha]} & {int(r.n_designs)} & {r.frr_pred_gp_median:.3f} & " + " & ".join(cells) + f" & {r.frr_test_median:.3f} \\\\")
out.append(r"\bottomrule\end{tabular}\end{table*}")

# S10: ablations (moved from the paper)
ab = pd.read_csv(f"{T}/T_ablation_paired.csv"); sens = pd.read_csv(f"{T}/T_ablation_sensitivity.csv")
aget = lambda v, m, a, c: ab[(ab.variant == v) & (ab.method == m) & np.isclose(ab.alpha, a)][c].iloc[0]
out.append(r"\begin{table}[!t]\centering\caption{Ablations on D1 at $\alpha=10^{-3}$ (Uncalibrated; Test FRR, Variant Versus Main Setting on the Same Splits; Wilcoxon $p$ Where Differences Exist)}\label{tab:s-abl}\footnotesize\setlength{\tabcolsep}{3pt}")
out.append(r"\begin{tabular}{@{}lccc@{}}\toprule Variant & Variant & Main & $p$ \\\midrule")
for lab, v, m in [("Sampled-point dominance", "abl_sample", "LR-P1-N2"), ("FAR range $[10^{-5},10^{-1}]$ only", "abl_region", "LR-P1-N2"),
                  ("$N=4$ (120-s limit)$^{\\ast}$", "abl_N", "LR-P1-N4"), ("$N=5$ (120-s limit)$^{\\ast}$", "abl_N", "LR-P1-N5"),
                  ("Grid spacing 0.01$^{\\ast}$", "sens_b001", "LR-P1-N2"), ("Grid spacing 0.05$^{\\ast}$", "sens_b005", "LR-P1-N2"),
                  ("60 bins$^{\\ast}$", "sens_nb60", "LR-P1-N2"), ("240 bins$^{\\ast}$", "sens_nb240", "LR-P1-N2")]:
    p = aget(v, m, 1e-3, "p_wilcoxon")
    out.append(f"{lab} & {aget(v, m, 1e-3, 'frr_test_variant'):.4f} & {aget(v, m, 1e-3, 'frr_test_main'):.4f} & {'--' if pd.isna(p) else f'{p:.3f}'} \\\\")
for tag, lab in [("frac03", "Training fraction 0.3$^{\\ast}$"), ("frac07", "Training fraction 0.7$^{\\ast}$")]:
    r = sens[(sens.tag == tag) & (sens.method == "LR-P1-N2") & np.isclose(sens.alpha, 1e-3)].iloc[0]
    out.append(f"{lab} & {r.frr_test:.4f} & -- & -- \\\\")
out.append(r"\bottomrule\multicolumn{4}{@{}l}{\footnotesize $^{\ast}$Five splits.}\end{tabular}\end{table}")

# S11: secondary comparisons (envelope families) with corrected CIs
st = pd.read_csv(f"{T}/T_rev_stats.csv"); sec = st[st.family == "secondary"]
labs = {"MONO": "MONO", "MONO-cont": "MONO-c", "DE-N2": "DE", "NLS-N2": "NLS", "MAXMONO-K3": "MAXMONO", "LR-P1-N3": "$N=3$", "LR-P2-N2": "P2", "LR-P1-N2-rel": "P1-rel"}
out.append(r"\begin{table*}[!t]\centering\caption{Envelope Families: Mean Test-FRR Difference (Proposed Minus Family) With 95\% Corrected-Resampled Confidence Interval, Subject-Bootstrap Calibration, Selection by Training FRR}\label{tab:s-sec}\footnotesize\setlength{\tabcolsep}{1.2pt}")
combos = [(ds, a) for ds in DS for a in sorted(sec[sec.dataset == ds].alpha.unique(), reverse=True)]
CID = 4 if "ijis" in os.environ.get("PAPER", "") else 3   # IJIS v05: CIs to 4 decimals (one CI bound is 0.0002)
fmt = lambda v, d: (f"{v:+.{d}f}".replace("-0." + "0" * d, "0." + "0" * d).replace("+0." + "0" * d, "0." + "0" * d)).replace("-", "$-$")
out.append(r"\begin{tabular}{@{}l" + "c" * len(combos) + r"@{}}\toprule & " + " & ".join(f"{DS[ds]}, {ALPHA[a]}" for ds, a in combos) + r" \\\midrule")
for m, lab in labs.items():
    cells = []
    for ds, a in combos:
        r = sec[(sec.dataset == ds) & np.isclose(sec.alpha, a) & (sec.method == m)]
        cells.append("--" if len(r) == 0 or pd.isna(r.mean_diff.iloc[0]) else
                     f"\\makecell{{{fmt(r.mean_diff.iloc[0], 4)}\\\\{{}}[{fmt(r.ci_lo.iloc[0], CID).lstrip('+')}, {fmt(r.ci_hi.iloc[0], CID).lstrip('+')}]}}")
    out.append(f"{lab} & " + " & ".join(cells) + r" \\")
out.append(r"\bottomrule\end{tabular}\end{table*}")

# S12: exact-dual root certification (E2v2), measured against the independent exact reference (E2ref)
rg = pd.read_csv(f"{T}/T_rev_rootgap.csv")
out.append(r"\begin{table}[!t]\centering\caption{Exact-Dual LR-BB on the Solver-Comparison Instances ($N=2$, 40 per Objective), Measured Against the Independent Exact Reference: Root Bound Equal to the Optimum (Within $10^{-4}$), Mean/Max Shortfall of the Root Bound, Root Incumbent Optimal, Nodes (Mean/Median/Max), Certified Runs; Last Column: Mean/Max Shortfall of the Subgradient Root Bound}\label{tab:s-root}\footnotesize\setlength{\tabcolsep}{2pt}")
out.append(r"\begin{tabular}{@{}lcccccc@{}}\toprule Obj. & \makecell{Bound\\$=$ opt.} & \makecell{Shortfall\\(\%)} & \makecell{Incumbent\\optimal} & Nodes & Certified & \makecell{Subgrad.\\shortfall (\%)} \\\midrule")
for _, r in rg.iterrows():
    out.append(f"{r.obj} & {int(r.bound_eq_opt)}/{int(r.n)} & {100 * r.bound_short_mean:.2f}/{100 * r.bound_short_max:.2f} & {int(r.inc_opt)}/{int(r.n)} & "
               f"{r.nodes_mean:.1f}/{r.nodes_median:.0f}/{int(r.nodes_max)} & {int(r.complete)}/{int(r.n)} & {100 * r.subgrad_short_mean:.0f}/{100 * r.subgrad_short_max:.0f} \\\\")
out.append(r"\bottomrule\end{tabular}\end{table}")

# S13: spoof analysis (if available)
import os
if os.path.exists(f"{T}/T_rev_spoof.csv"):
    sp = pd.read_csv(f"{T}/T_rev_spoof.csv")
    ml = [("LR-P1-N2", "Proposed"), ("S1-Marcialis", "Marcialis"), ("S2-Symmetric", "Symmetric"), ("S4-Direct", "Direct"), ("S3-SPRT", "SPRT"),
          ("P0-Parallel", "Sum"), ("P1-LLR", "LLR"), ("P2-LogReg", "LogReg")]
    out.append(r"\begin{table*}[!t]\centering\caption{Acceptance Rate of an Impostor Presenting a Perfect Spoof of One Modality (Score Drawn From the Genuine Test Scores), Selected Designs at $\alpha=10^{-3}$: Worst Case / Mean Over the Modalities of the Chain, Averaged Over Splits. In D3 Both Matchers Process the Same Face Image, so the D3 Row Describes a Spoof That Deceives Only One of the Two Algorithms}\label{tab:s-spoof}\footnotesize\setlength{\tabcolsep}{3pt}")
    out.append(r"\begin{tabular}{@{}l" + "c" * len(ml) + r"@{}}\toprule Set & " + " & ".join(l for _, l in ml) + r" \\\midrule")
    for ds in DS:
        cells = []
        for m, _ in ml:
            r = sp[(sp.dataset == ds) & (sp.method == m)]
            cells.append("--" if len(r) == 0 else f"{r.spoof_max.iloc[0]:.3f} / {r.spoof_mean.iloc[0]:.3f}")
        out.append(f"{DS[ds]} & " + " & ".join(cells) + r" \\")
    out.append(r"\bottomrule\end{tabular}\end{table*}")

# S14: realized FAR and stage use of all methods (selected designs, subject-bootstrap calibration)
sy = pd.read_csv(f"{T}/T_rev_system.csv")
ml2 = [("LR-P1-N2", "Proposed"), ("S1-Marcialis", "Marcialis"), ("S4-Direct", "Direct"), ("S3-SPRT", "SPRT"), ("P0-Parallel", "Sum"), ("P1-LLR", "LLR"), ("P2-LogReg", "LogReg")]
out.append(r"\begin{table*}[!t]\centering\caption{Selected Designs Under Subject-Bootstrap Calibration: Mean Test FAR$/\alpha$ and Stages per Genuine / Impostor Claim (Means Over Ten Splits; Parallel Fusion Acquires All Modalities)}\label{tab:s-farstage}\footnotesize\setlength{\tabcolsep}{2.5pt}")
out.append(r"\begin{tabular}{@{}ll" + "c" * len(ml2) + r"@{}}\toprule Set & $\alpha$ & " + " & ".join(l for _, l in ml2) + r" \\\midrule")
for ds in DS:
    for a in sorted(sy[sy.dataset == ds].alpha.unique(), reverse=True):
        cells = []
        for m, _ in ml2:
            r = sy[(sy.dataset == ds) & np.isclose(sy.alpha, a) & (sy.method == m)]
            if len(r) == 0 or r.n_seeds.iloc[0] < 10: cells.append("--"); continue
            r = r.iloc[0]
            cells.append(f"{r.far_over_alpha:.2f}; {r.stages_gen:.2f}/{r.stages_imp:.2f}" if not m.startswith("P") else f"{r.far_over_alpha:.2f}")
        out.append(f"{DS[ds]} & {ALPHA[a]} & " + " & ".join(cells) + r" \\")
out.append(r"\bottomrule\end{tabular}\end{table*}")

# S15: calibration diagnostic (bootstrap spread vs. shift between halves)
import os
if os.path.exists(f"{T}/T_rev_calib_diag.csv"):
    cd = pd.read_csv(f"{T}/T_rev_calib_diag.csv"); pr_ = cd[cd.method == "LR-P1-N2"]
    out.append(r"\begin{table}[!t]\centering\caption{Calibration Diagnostic for the Selected Proposed Designs (Means Over Ten Splits, in Units of $\alpha$): Training and Test FAR, Mean Shift (Test $-$ Training) and Its Split-to-Split SD Divided by $\sqrt2$, Bootstrap SD of the Training FAR at the Deployed Thresholds; Last Column: Range of the Mean Shift Over the Other Six Methods}\label{tab:s-calib}\footnotesize\setlength{\tabcolsep}{2.2pt}")
    SPR_ = "ijis" in os.environ.get("PAPER", "")
    tcol = lambda d: d.shift_mean / (d.shift_sd * np.sqrt(1 + 1 / 10))   # corrected resampled t (K = 10, n_test/n_train = 1)
    if SPR_:
        out.append(r"\begin{tabular}{@{}llcccccccc@{}}\toprule Set & $\alpha$ & Train & Test & Shift & \makecell{Shift SD\\$/\sqrt2$} & \makecell{Boot.\\SD} & $t$ & \makecell{Shift,\\others} & \makecell{$|t|$,\\others} \\\midrule")
    else:
        out.append(r"\begin{tabular}{@{}llcccccc@{}}\toprule Set & $\alpha$ & Train & Test & Shift & \makecell{Shift SD\\$/\sqrt2$} & \makecell{Boot.\\SD} & \makecell{Shift,\\others} \\\midrule")
    for ds in DS:
        for a in sorted(pr_[pr_.dataset == ds].alpha.unique(), reverse=True):
            r = pr_[(pr_.dataset == ds) & np.isclose(pr_.alpha, a)].iloc[0]
            o = cd[(cd.dataset == ds) & np.isclose(cd.alpha, a) & (cd.method != "LR-P1-N2")].shift_mean
            sg = lambda v: f"{v:+.2f}".replace("-", "$-$")
            if SPR_:
                oo = cd[(cd.dataset == ds) & np.isclose(cd.alpha, a) & (cd.method != "LR-P1-N2")]
                out.append(f"{DS[ds]} & {ALPHA[a]} & {r.far_train_over_alpha:.2f} & {r.far_test_over_alpha:.2f} & {sg(r.shift_mean)} & {r.shift_sd_over_sqrt2:.2f} & {r.boot_sd_mean:.2f} & {sg(tcol(r))} & {sg(o.min())} to {sg(o.max())} & {tcol(oo).abs().max():.2f} \\\\")
            else:
                out.append(f"{DS[ds]} & {ALPHA[a]} & {r.far_train_over_alpha:.2f} & {r.far_test_over_alpha:.2f} & {sg(r.shift_mean)} & {r.shift_sd_over_sqrt2:.2f} & {r.boot_sd_mean:.2f} & {sg(o.min())} to {sg(o.max())} \\\\")
    out.append(r"\bottomrule\end{tabular}\end{table}")

# S16: stage-constrained designs (E8)
import glob as _g
fs = sorted(_g.glob("../results/E8/cost_*.csv"))
if fs:
    e8 = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    out.append(r"\begin{table*}[!t]\centering\caption{Designs Under a Cap $\kappa$ on the Expected Number of Stages per Impostor Claim ($\alpha=10^{-3}$, Most Frequently Selected Order, Subject-Bootstrap Calibration): Test FRR, Test Stages per Impostor Claim, and Worst-Case Single-Modality Spoof Acceptance (Means Over Ten Splits). GP: Posynomial Constraint in (3); SPRT: Boundaries Chosen on the Training Half Under the Same Cap}\label{tab:s-cost}\footnotesize\setlength{\tabcolsep}{3pt}")
    out.append(r"\begin{tabular}{@{}llcccccc@{}}\toprule Set (order) & $\kappa$ & \multicolumn{3}{c}{GP with stage cap} & \multicolumn{3}{c}{SPRT with stage cap} \\\cmidrule(lr){3-5}\cmidrule(lr){6-8} & & FRR & Stages & Spoof & FRR & Stages & Spoof \\\midrule")
    for ds in DS:
        d = e8[e8.dataset == ds]
        if len(d) == 0: continue
        for k in sorted(d.param.unique(), reverse=True):
            cells = []
            for meth in ["GP stage cap", "SPRT stage cap"]:
                g = d[(d.method == meth) & np.isclose(d.param, k)]
                ok = g[g.feasible == True]
                if len(ok) < len(g) or len(g) == 0: cells += ["--"] * 3; continue
                cells += [f"{ok.frr_test.mean():.4f}", f"{ok.stages_imp.mean():.2f}", f"{ok.spoof_worst.mean():.2f}"]
            lab = f"{DS[ds]} ({d.order.iloc[0].replace('>', '$>$').replace('_', '')})" if k == np.inf else ""
            out.append(f"{lab} & {'none' if k == np.inf else f'{k:.1f}'} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule\end{tabular}\end{table*}"

# S17: refit of the system-level envelopes with the exact-dual LR-BB
fs = sorted(_g.glob("../results/E1xd/xd_*.csv"))
if fs:
    xd = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    out.append(r"\begin{table}[!t]\centering\caption{Envelopes of the System Experiments (Fitted by the Subgradient LR-BB of the First Version) Refitted With the Exact-Dual LR-BB: Number of Envelopes, Share With Identical Curves (Largest Relative Difference $<10^{-6}$), and Largest Relative Curve Difference}\label{tab:s-refit}\footnotesize\setlength{\tabcolsep}{3pt}")
    out.append(r"\begin{tabular}{@{}lccc@{}}\toprule Envelope & $n$ & Identical & Largest difference \\\midrule")
    for meth in ["LR-P1-N2", "LR-P1-N3", "LR-P2-N2", "LR-P1-N2-rel", "LR-P2-N2-rel"]:
        g = xd[xd.method == meth]
        if len(g) == 0: continue
        out.append(f"{meth.replace('LR-', '')} & {len(g)} & {100 * (g.max_rel_curve_diff < 1e-6).mean():.0f}\\% & {100 * g.max_rel_curve_diff.max():.2f}\\% \\\\")
    out.append(r"\bottomrule\end{tabular}\end{table}")

# ---- IJIS v02: matched-FAR comparison, fresh-split replication, cross-fitted calibration (Springer version only)
_PAPER = os.environ.get("PAPER", "../paper"); NEW_CAP = {}
_LO, _HI = "$^{\\mathrm{a}}$", "$^{\\mathrm{b}}$"
_ML = [("LR-P1-N2", "Proposed"), ("HYP", "HYP"), ("S1-Marcialis", "Marcialis"), ("S2-Symmetric", "Symmetric"), ("S4-Direct", "Direct"),
       ("S3-SPRT", "SPRT"), ("P0-Parallel", "Sum"), ("P1-LLR", "LLR"), ("P2-LogReg", "LogReg")]
_DS4 = {"fing_x_face": "D1", "fing_x_fing": "D2", "face_x_face": "D3", "lfw_x_fing": "D4"}
def _grid(sysr, st, val, dss, count=True):
    rows = []
    for ds in dss:
        als = sorted(sysr[sysr.dataset == ds].alpha.unique(), reverse=True)
        for k, a in enumerate(als):
            cells = [_DS4[ds] if k == 0 else "", ALPHA[a]]
            for m, _ in _ML:
                r = sysr[(sysr.dataset == ds) & np.isclose(sysr.alpha, a) & (sysr.method == m)]
                if len(r) == 0 or pd.isna(r[val].iloc[0]): cells.append("--"); continue
                r = r.iloc[0]; txt = f"{r[val]:.4f}"
                s_ = st[(st.dataset == ds) & np.isclose(st.alpha, a) & (st.method == m)]
                if m != "LR-P1-N2" and len(s_) and pd.notna(s_.p_holm.iloc[0]) and s_.p_holm.iloc[0] < 0.05:
                    txt += _LO if s_.mean_diff.iloc[0] < 0 else _HI
                n_ = int(r.get("n_seeds", r.get("n", 10)))
                if count and "far_ok" in r and pd.notna(r.far_ok):
                    txt += " (" + f"{r.far_ok * n_:.2f}".rstrip("0").rstrip(".") + ")"
                if ("n_seeds" in sysr) and n_ < sysr[sysr.dataset == ds].n_seeds.max(): txt += "$^{\\mathrm{c}}$"   # per subset
                cells.append(txt)
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    return rows[:-1]
_HDR = r"\begin{tabular}{@{}ll" + "c" * len(_ML) + r"@{}}\toprule Set & $\alpha$ & " + " & ".join(l for _, l in _ML) + r" \\\midrule"
_NOTE = lambda extra: (r"\par\smallskip\parbox{\textwidth}{\footnotesize $^{\mathrm{a}}$Proposed design lower and $^{\mathrm{b}}$proposed design higher "
                       r"(corrected resampled $t$-test, Holm-adjusted $p<0.05$ within each setting)." + extra + "}")
if "ijis" in _PAPER and os.path.exists(f"{T}/T_rev_matchedfar_system.csv"):
    mf = pd.read_csv(f"{T}/T_rev_matchedfar_system.csv").rename(columns={"n": "n_seeds"}); mfs = pd.read_csv(f"{T}/T_rev_matchedfar_stats.csv")
    out.append(r"\begin{table*}[!t]\centering\caption{CAPTION}\label{tab:s-matched}\scriptsize\setlength{\tabcolsep}{3pt}" + _HDR)
    out += _grid(mf, mfs, "frr_matched", ["fing_x_face", "fing_x_fing", "face_x_face"], count=False)
    out.append(r"\bottomrule\end{tabular}" + _NOTE(" --, no design reaching the matched FAR.") + r"\end{table*}")
    NEW_CAP["tab:s-matched"] = (r"Matched-FAR comparison (D1--D3, original splits): test FRR of the selected designs after the final threshold is re-set on the test half "
                                r"to the most permissive value with test FAR $\le\alpha$ (earlier stage thresholds kept; oracle operating point), mean over ten splits")
if "ijis" in _PAPER and os.path.exists(f"{T}/T_fresh_system.csv"):
    fsy = pd.read_csv(f"{T}/T_fresh_system.csv"); fst = pd.read_csv(f"{T}/T_fresh_stats.csv")
    for cal, lab, dss, cap in [("boot", "tab:s-fresh", ["fing_x_face", "fing_x_fing", "face_x_face"],
                                r"Replication on 20 further splits of D1--D3 (seeds 10--29; new halvings of the same subjects) with the calibration of the paper: test FRR of the selected designs (mean over the new splits) and, in parentheses, number of splits with test FAR $\le\alpha$ (tie-averaged)"),
                               ("xfit", "tab:s-xfit", ["fing_x_face", "fing_x_fing", "face_x_face", "lfw_x_fing"],
                                r"Held-out calibration (20 new splits of D1--D3, ten splits of D4): design, stage thresholds and order selection on one half of the training subjects, final threshold by the subject bootstrap on the other half; test FRR (mean) and, in parentheses, number of splits with test FAR $\le\alpha$ (tie-averaged)")]:
        a_ = fsy[fsy.calib == cal]; b_ = fst[fst.calib == cal]
        out.append(r"\begin{table*}[!t]\centering\caption{CAPTION}\label{" + lab + r"}\scriptsize\setlength{\tabcolsep}{2.5pt}" + _HDR)
        out += _grid(a_, b_, "frr_test", dss)
        out.append(r"\bottomrule\end{tabular}" + _NOTE(" $^{\mathrm{c}}$Mean over the splits with a feasible design only. --, no feasible design.") + r"\end{table*}")
        NEW_CAP[lab] = cap
if "ijis" in _PAPER and os.path.exists(f"{T}/T_fresh_matchedfar_system.csv"):   # IJIS v02: matched FAR on the new splits and D4
    mfn = pd.read_csv(f"{T}/T_fresh_matchedfar_system.csv").assign(n_seeds=lambda d: d.n_feasible_all)
    mfns = pd.read_csv(f"{T}/T_fresh_matchedfar_stats.csv")
    out.append(r"\begin{table*}[!t]\centering\caption{CAPTION}\label{tab:s-matchednew}\scriptsize\setlength{\tabcolsep}{3pt}" + _HDR)
    out += _grid(mfn, mfns, "frr_matched", ["fing_x_face", "fing_x_fing", "face_x_face", "lfw_x_fing"], count=False)
    out.append(r"\bottomrule\end{tabular}" + _NOTE(" $^{\mathrm{c}}$Mean over the splits in which the matched FAR could be reached. --, no design reaching the matched FAR.") + r"\end{table*}")
    NEW_CAP["tab:s-matchednew"] = (r"Matched-FAR comparison on the 20 new splits of D1--D3 and the ten splits of D4 (calibration of the paper): test FRR of the selected designs "
                                   r"after the final threshold is re-set on the test half to the most permissive value with test FAR $\le\alpha$ (earlier stage thresholds kept; oracle operating point), mean over the splits")
if "ijis" in _PAPER and os.path.exists(f"{T}/T_mlp_system.csv"):   # IJIS v04: MLP fusion baseline (secondary, unadjusted p)
    msy = pd.read_csv(f"{T}/T_mlp_system.csv"); mst = pd.read_csv(f"{T}/T_mlp_stats.csv")
    out.append(r"\begin{table*}[!t]\centering\caption{CAPTION}\label{tab:s-mlp}\scriptsize\setlength{\tabcolsep}{4pt}"
               r"\begin{tabular}{@{}lllccccc@{}}\toprule Set & Splits & Calibration & $\alpha$ & MLP & Proposed & LLR & LogReg \\\midrule")
    _SS = [("original 0-9", "boot", "0--9", "bootstrap"), ("fresh", "boot", "10--29", "bootstrap"), ("fresh", "xfit", "10--29", "held-out"),
           ("D4 0-9", "boot", "0--9", "bootstrap"), ("D4 0-9", "xfit", "0--9", "held-out")]
    for ds in ["fing_x_face", "fing_x_fing", "face_x_face", "lfw_x_fing"]:
        first = True
        for ss, cal, slab, clab in _SS:
            g = msy[(msy.dataset == ds) & (msy.split_set == ss) & (msy.calib == cal)]
            for k, a in enumerate(sorted(g.alpha.unique(), reverse=True)):
                r = g[np.isclose(g.alpha, a)].iloc[0]
                cells = [_DS4[ds] if first else "", slab if k == 0 else "", clab if k == 0 else "", ALPHA[a],
                         f"{r.frr_test:.4f} (" + f"{r.far_ok * r.n:.2f}".rstrip("0").rstrip(".") + f"/{int(r.n)})"]
                first = False
                for other in ["LR-P1-N2", "P1-LLR", "P2-LogReg"]:
                    q = mst[(mst.dataset == ds) & (mst.split_set == ss) & (mst.calib == cal) & np.isclose(mst.alpha, a) & (mst.other == other)]
                    if len(q) == 0: cells.append("--"); continue
                    q = q.iloc[0]; txt = f"{q.frr_other:.4f}"
                    if q.p_unadj < 0.05: txt += _LO if q.mean_diff < 0 else _HI
                    cells.append(txt)
                out.append(" & ".join(cells) + r" \\")
        out.append(r"\midrule")
    out[-1] = (r"\bottomrule\end{tabular}\par\smallskip\parbox{\textwidth}{\footnotesize $^{\mathrm{a}}$MLP lower and $^{\mathrm{b}}$MLP higher "
               r"(marks on the value of the compared method; corrected resampled $t$-test, unadjusted $p<0.05$; secondary comparison outside the Holm families).}\end{table*}")
    NEW_CAP["tab:s-mlp"] = (r"Parallel fusion by a multilayer perceptron (MLP; two hidden layers of 16 rectified linear units; secondary baseline): "
                            r"test FRR (mean over the splits) with, in parentheses, the number of splits with test FAR $\le\alpha$, and the test FRR of the "
                            r"selected proposed design (order selected by training FRR, ties averaged), LLR fusion, and logistic regression on the same splits and with the same calibration")
text = "\n".join(out) + "\n"
PAPER = os.environ.get("PAPER", "../paper")
if "ijis" in PAPER:
    # Springer (IJIS) Online Resource 1: tables numbered in order of first citation, sentence-case captions, footnote
    # letters instead of asterisks; the spoof and stage-cap tables are in the main text (Tables 5 and 6) and omitted here.
    import re
    CAP = {
        "tab:s-root": r"Exact-dual LR-BB on the solver-comparison instances ($N=2$, 40 per objective), measured against the independent exact reference: root bound equal to the optimum (within $10^{-4}$), mean/max shortfall of the root bound, root incumbent optimal, nodes (mean/median/max), certified runs; last column: mean/max shortfall of the subgradient root bound",
        "tab:s-calib": r"Calibration diagnostic for the selected proposed designs (means over ten splits, in units of $\alpha$): training and test FAR, mean shift (test $-$ training) and its split-to-split SD divided by $\sqrt2$, bootstrap SD of the training FAR at the deployed thresholds, and corrected resampled $t$ of the shift (mean divided by $\sqrt{1.1}$ times the split-to-split SD); last two columns: range of the mean shift and largest $|t|$ over the other six methods",
        "tab:s-plain": r"Uncalibrated designs (thresholds at the point estimate of the training FAR, as in the prior GP design): test FRR (mean over ten splits) and, in parentheses, share of splits with test FAR $\le\alpha$",
        "tab:s-cp": r"Designs with pair-level Clopper--Pearson calibration (GP designs selected by predicted FRR, rules by training FRR, first tie kept): test FRR",
        "tab:s-cons": r"Subject-bootstrap calibration: share of proposed designs with test FAR $\le\alpha$, median test FAR$/\alpha$, and share with test FRR $\le$ prediction, by chain type",
        "tab:s-decomp": r"Per-design decomposition of the predicted system FRR (uncalibrated proposed designs, all orders): median [interquartile range] of each successive difference",
        "tab:s-sec": r"Envelope families: mean test-FRR difference (proposed minus family) with 95\% corrected-resampled confidence interval, subject-bootstrap calibration, selection by training FRR",
        "tab:s-fit": r"Envelope fitting on the training halves: ratio of the sum of squared errors (SSE) to that of LR-BB P1 (geometric mean), mean $\log_{10}$ overestimation over all corners, share of test corners above the envelope, and fitting time (s, including constraint generation; the LR-BB envelopes of this table were fitted by the subgradient LR-BB of the preliminary implementation and refitted by the exact-dual LR-BB for the calibrated designs, Table~\ref{tab:s-refit})",
        "tab:s-scal": r"Scalability (seed 0, P1, single core, 600-s limit per solver). Enumeration times marked $^{e}$ are extrapolated from the enumerated fraction",
        "tab:s-reject": r"Stage FAR at the accept and reject thresholds of the uncalibrated proposed GP designs (all multi-stage orders, ten splits)",
        "tab:s-abl": r"Ablations on D1 at $\alpha=10^{-3}$ (uncalibrated; test FRR, variant versus main setting on the same splits; Wilcoxon $p$ where differences exist)",
        "tab:s-noise": r"Robustness of D1 designs to additive Gaussian test-score noise (SD = 5\% or 10\% of the impostor-score SD; uncalibrated): test FAR / FRR (means over all orders and splits)",
        "tab:s-farstage": r"Selected designs under subject-bootstrap calibration: mean test FAR$/\alpha$ and stages per genuine / impostor claim (means over ten splits; parallel fusion acquires all modalities)",
        "tab:s-refit": r"Envelopes fitted by the subgradient LR-BB of the preliminary implementation (used in the uncalibrated and Clopper--Pearson analyses) compared with their refit by the exact-dual LR-BB (used in the calibrated designs): number of envelopes, share with identical curves (largest relative difference $<10^{-6}$), and largest relative curve difference",
    }
    CAP.update(NEW_CAP)
    blocks = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, re.S)
    rest = text
    for b_ in blocks: rest = rest.replace(b_, "")
    assert rest.strip() == "", "text outside table environments"
    by = {re.search(r"\\label\{(tab:s-[a-z]+)\}", b).group(1): b for b in blocks}
    assert set(by) == set(CAP) | {"tab:s-spoof", "tab:s-cost"}, sorted(by)
    new = []
    for lab in CAP:
        b = by[lab]; i = b.index("\\caption{") + len("\\caption{"); j = b.index("}\\label{" + lab + "}")
        b = b[:i] + CAP[lab] + b[j:]
        b = b.replace("$^{\\ast}$", "$^{\\mathrm{a}}$")
        if lab == "tab:s-calib": b = b.replace("\\begin{table}[!t]", "\\begin{table*}[!t]").replace("\\end{table}", "\\end{table*}").replace("\\setlength{\\tabcolsep}{2.2pt}", "\\setlength{\\tabcolsep}{5pt}")
        if lab == "tab:s-plain": b = b.replace(" & Parallel \\\\", " & Sum \\\\")   # column is sum fusion (P0-Parallel)
        if lab == "tab:s-sec": b = b.replace("\\footnotesize\\setlength{\\tabcolsep}{1.2pt}", "\\scriptsize\\setlength{\\tabcolsep}{1.5pt}")
        new.append(b)
    text = "\n".join(new) + "\n"
open(f"{PAPER}/supp_tables.tex", "w").write(text)
print(f"written {PAPER}/supp_tables.tex")
