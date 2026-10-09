"""Generate the LaTeX bodies of the main-paper tables directly from results/tables (no manual transcription).
Outputs $PAPER/tables/*.tex (default ../paper, the IEEE TIFS version; PAPER=../paper_ijis gives the Springer IJIS
version, which marks significance by superscript letters and also has the spoof and stage-cap tables in the main text)"""
import glob, os, numpy as np, pandas as pd
PAPER = os.environ.get("PAPER", "../paper"); SPRINGER = "ijis" in PAPER
T = "../results/tables"; R = "../results"; OUT = f"{PAPER}/tables"; os.makedirs(OUT, exist_ok=True)
MK_LO, MK_HI, MK_PART = (("$^{\\mathrm{a}}$", "$^{\\mathrm{b}}$", "$^{\\mathrm{c}}$") if SPRINGER
                         else ("$^{\\dagger}$", "$^{\\ddagger}$", "$^{\\S}$"))
DS = {"fing_x_face": "D1", "fing_x_fing": "D2", "face_x_face": "D3"}
AL = {1e-2: "$10^{-2}$", 1e-3: "$10^{-3}$", 1e-4: "$10^{-4}$"}

# ------------------------------------------------------------------ Table II: solvers (E2 + E2v2, N = 2), against the independent reference
e2 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2/e2_*.csv")], ignore_index=True)
v2 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2v2/e2v2_*_N2.csv")], ignore_index=True)
ref = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2ref/ref_*.csv")], ignore_index=True)
KEY = ["dataset", "seed", "matcher", "obj"]
e2 = e2.merge(ref[KEY + ["value"]].rename(columns={"value": "opt"}), on=KEY)
v2 = v2.merge(ref[KEY + ["value"]].rename(columns={"value": "opt"}), on=KEY)
for d_ in (e2, v2): d_["excess"] = (d_.value - d_.opt) / d_.opt
e2["short"] = (e2.opt - e2.lb) / e2.opt                                   # root bound below the optimum (subgradient rows)
v2["short"] = (v2.opt - v2.root_lb) / v2.opt if "root_lb" in v2 else np.nan
def pm(v, pct=True, d=1):
    if len(v) == 0: return "--"
    v = v * (100 if pct else 1); f = lambda x: (f"{x:.0f}" if abs(x) >= 10 else f"{x:.{d}f}").replace("-0.0", "0.0")
    return f"{f(v.mean())}/{f(v.max())}"
def trange(t):
    t = t.groupby(level=0).mean() if isinstance(t.index, pd.MultiIndex) else t
    fmt = "%.1f" if t.min() >= 1 else "%.2f"
    lo, hi = fmt % t.min(), fmt % t.max()
    return lo if lo == hi else f"{lo}--{hi}"
lines = []
def row(label, df, short):
    cells = [label] + ([pm(df[df.obj == o].excess.clip(lower=-1)) for o in ("P1", "P2")] if SPRINGER else [pm(df.excess.clip(lower=-1))])
    for obj in ["P1", "P2"]:
        d = df[df.obj == obj]
        cells.append(pm(d.short) if (short and len(d)) else "--")
    for obj in ["P1", "P2"]:
        d = df[df.obj == obj]; cells.append(trange(d.groupby("dataset").time.mean()) if len(d) else "--")
    return " & ".join(cells) + r" \\"
lines.append(row("Enumeration (v1)", e2[e2.method == "ENUM"], False))
lines.append(row(r"Subgrad. LR, root~\cite{yeh2023}", e2[e2.method == "LR-root (no OBBT)"], True))
bb = e2[e2.method == "LR-BB + OBBT"].drop(columns="short").merge(e2[e2.method == "LR-root + OBBT"][KEY + ["short"]], on=KEY)
lines.append(row("Subgrad. LR-BB (v1)", bb, True))
lines.append(row(r"\textbf{Exact-dual LR-BB}", v2[v2.method == "LR-BB exact dual"], True))
lines.append(row("MILP (HiGHS)", v2[v2.method == "MILP (HiGHS)"], False))
if SPRINGER and glob.glob(f"{R}/E2miqp/miqp_*_N2.csv"):                      # IJIS v02: MIQP by outer approximation (P1 only)
    mq = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2miqp/miqp_*_N2.csv")], ignore_index=True)
    mq = mq.merge(ref[KEY + ["value"]].rename(columns={"value": "opt"}), on=KEY); mq["excess"] = (mq.value - mq.opt) / mq.opt
    assert len(mq) == 40 and (mq.status == "optimal").all()
    lines.append(row(r"MIQP-OA (HiGHS)~\cite{duran1986}", mq, False))
de = e2[e2.method == "DE"]
dex = ([f"{100 * de[de.obj == o].excess.mean():.1f}/{100 * de[de.obj == o].excess.max():.1f}".replace("/-0.0", "/0.0") for o in ("P1", "P2")] if SPRINGER
       else [f"{100 * de.excess.mean():.1f}/{100 * de.excess.max():.1f}"])
lines.append(" & ".join(["Diff. evol."] + dex + ["--", "--",
                         trange(de[de.obj == "P1"].groupby("dataset").time.mean()), trange(de[de.obj == "P2"].groupby("dataset").time.mean())]) + r" \\")
open(f"{OUT}/tab_solver_body.tex", "w").write("\n".join(lines) + "\n")
ex = v2[v2.method == "LR-BB exact dual"].copy(); ex["inc_excess"] = (ex.root_ub - ex.opt) / ex.opt
root = ex.groupby("obj").apply(lambda d: pd.Series({
    "n": len(d), "bound_eq_opt": int((d.short <= 1e-4).sum()), "bound_short_mean": d.short.mean(), "bound_short_max": d.short.max(),
    "inc_opt": int((d.inc_excess <= 1e-6).sum()), "inc_excess_max": d.inc_excess.max(), "nodes_mean": d.nodes.mean(),
    "nodes_median": d.nodes.median(), "nodes_max": d.nodes.max(), "complete": int(d.complete.sum()),
    "final_excess_max": d.excess.abs().max()}))
sg = e2[e2.method == "LR-root (no OBBT)"].groupby("obj").short.agg(["mean", "max"]).rename(columns=lambda c: "subgrad_short_" + c)
root = root.join(sg); root["source"] = "results/E2v2, results/E2 and results/E2ref (independent exact reference)"
root.to_csv(f"{T}/T_rev_rootgap.csv")
print(open(f"{OUT}/tab_solver_body.tex").read()); print(root.T)

# ------------------------------------------------------------------ Table IV: system results (tie-averaged selection)
sysr = pd.read_csv(f"{T}/T_rev_system.csv"); st = pd.read_csv(f"{T}/T_rev_stats.csv")
cols = [("LR-P1-N2", None), ("HYP", "HYP"), ("S1-Marcialis", "Marc."), ("S2-Symmetric", "Symm."), ("S4-Direct", "Direct"),
        ("S3-SPRT", "SPRT"), ("P0-Parallel", "Sum"), ("P1-LLR", "LLR"), ("P2-LogReg", "LogReg")]
def sys_rows(sysr, st, ds, label):
    out = []
    als = sorted(sysr[sysr.dataset == ds].alpha.unique(), reverse=True)
    for k, a in enumerate(als):
        cells = [label if k == 0 else "", AL[a]]
        for m, _ in cols:
            r = sysr[(sysr.dataset == ds) & np.isclose(sysr.alpha, a) & (sysr.method == m)]
            if len(r) == 0 or r.n_seeds.iloc[0] < 10:
                if len(r) and m == "HYP" and r.n_seeds.iloc[0] > 0:
                    s_ = st[(st.dataset == ds) & np.isclose(st.alpha, a) & (st.method == m) & (st.family == "primary")]
                    mark = (MK_LO if s_.mean_diff.iloc[0] < 0 else MK_HI) if (len(s_) and pd.notna(s_.p_holm.iloc[0]) and s_.p_holm.iloc[0] < 0.05) else ""
                    cells.append(f"{r.frr_test.iloc[0]:.4f}{mark}{MK_PART}".replace("}}$$^{\\mathrm{", ",")); continue
                cells.append("--"); continue
            r = r.iloc[0]; txt = f"{r.frr_test:.4f}"
            if m == "LR-P1-N2": txt = f"{r.frr_test:.4f}" if SPRINGER else f"\\textbf{{{r.frr_test:.4f}}}"
            else:
                s = st[(st.dataset == ds) & np.isclose(st.alpha, a) & (st.method == m) & (st.family == "primary")]
                if len(s) and pd.notna(s.p_holm.iloc[0]) and s.p_holm.iloc[0] < 0.05:
                    txt += MK_LO if s.mean_diff.iloc[0] < 0 else MK_HI
            k_ok = 10 * r.far_ok
            if pd.notna(r.far_ok):
                txt += (" (" + f"{k_ok:.2f}".rstrip("0").rstrip(".") + ")") if (SPRINGER and abs(k_ok - round(k_ok)) > 1e-9) else f" ({k_ok:.0f})"
            cells.append(txt)
        out.append(" & ".join(cells) + r" \\")
    return out
lines = []
for ds in DS:
    lines += sys_rows(sysr, st, ds, DS[ds]); lines.append(r"\midrule")
FRESH = SPRINGER and os.path.exists(f"{T}/T_fresh_system.csv")
if FRESH:   # IJIS v02: subset D4 (contemporary face matcher), ten splits, same calibration
    fsys = pd.read_csv(f"{T}/T_fresh_system.csv"); fsys = fsys[fsys.calib == "boot"]
    fst = pd.read_csv(f"{T}/T_fresh_stats.csv"); fst = fst[fst.calib == "boot"].assign(family="primary")
    lines += sys_rows(fsys, fst, "lfw_x_fing", "D4"); lines.append(r"\midrule")
lines = lines[:-1]
stg = []
for m, _ in cols:
    r = sysr[sysr.method == m]
    if FRESH: r = pd.concat([r, fsys[(fsys.dataset == "lfw_x_fing") & (fsys.method == m)]])
    stg.append(f"{r.stages_gen.min():.2f}--{r.stages_gen.max():.2f}" if len(r) else "--")
lines.append(r"\midrule")
lines.append((r"\multicolumn{2}{@{}l}{\makecell[l]{Stages per\\genuine claim}} & " if SPRINGER else r"\multicolumn{2}{@{}l}{Stages per genuine claim} & ") + " & ".join(stg) + r" \\")
open(f"{OUT}/tab_system_body.tex", "w").write("\n".join(lines) + "\n")
print(open(f"{OUT}/tab_system_body.tex").read())

# ------------------------------------------------------------------ Table V: per-design decomposition (medians)
dc = pd.read_csv(f"{T}/T_rev_decomposition.csv")
rs = {ds: dc[(dc.dataset == ds) & np.isclose(dc.alpha, 1e-3)].iloc[0] for ds in DS}
lines = [" & ".join(["Designs"] + [f"{int(rs[ds].n_designs)}" for ds in DS]) + r" \\", r"\midrule",
         " & ".join(["GP objective $\\overline{\\frr}$"] + [f"{rs[ds].frr_pred_gp_median:.3f}" for ds in DS]) + r" \\"]
for step, lab in [("series bound", "Series bound"), ("envelope", "Envelope overestimation"), ("dependence", "Dependence"),
                  ("generalization", "Generalization")]:
    lines.append(" & ".join([lab] + [f"{rs[ds][step + '_median']:+.3f}".replace("-0.000", "0.000").replace("+0.000", "0.000") for ds in DS]) + r" \\")
lines.append(" & ".join(["Test FRR"] + [f"{rs[ds].frr_test_median:.3f}" for ds in DS]) + r" \\")
open(f"{OUT}/tab_decomp_body.tex", "w").write("\n".join(lines) + "\n")
print(open(f"{OUT}/tab_decomp_body.tex").read())

# ------------------------------------------------------------------ IJIS only: spoof and stage-cap tables in the main text
if SPRINGER:
    sp = pd.read_csv(f"{T}/T_rev_spoof_trait.csv")
    ml = ["LR-P1-N2", "S1-Marcialis", "S2-Symmetric", "S4-Direct", "S3-SPRT", "P0-Parallel", "P1-LLR", "P2-LogReg"]
    lines = []
    for ds in list(DS) + (["lfw_x_fing"] if (sp.dataset == "lfw_x_fing").any() else []):   # IJIS v02: D4 row
        cells = []
        for m in ml:
            r = sp[(sp.dataset == ds) & (sp.method == m)]
            cells.append("--" if len(r) == 0 else f"{r.spoof_trait_max.iloc[0]:.3f} / {r.spoof_trait_mean.iloc[0]:.3f}")
        lines.append(f"{DS.get(ds, 'D4')} & " + " & ".join(cells) + r" \\")
    open(f"{OUT}/tab_spoof_body.tex", "w").write("\n".join(lines) + "\n")
    e8 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/E8/cost_*.csv"))], ignore_index=True)
    stt = pd.read_csv(f"{T}/T_rev_cost_spoof_trait.csv")
    e8 = e8.merge(stt[["dataset", "seed", "method", "param", "spoof_trait_worst"]], on=["dataset", "seed", "method", "param"], how="left")
    lines = []
    for ds in DS:
        d = e8[e8.dataset == ds]
        for k in sorted(d.param.unique(), reverse=True):
            cells = []
            for meth in ["GP stage cap", "SPRT stage cap"]:
                g = d[(d.method == meth) & np.isclose(d.param, k)]; ok = g[g.feasible == True]
                if len(ok) < len(g) or len(g) == 0: cells += ["--"] * 3; continue
                cells += [f"{ok.frr_test.mean():.4f}", f"{ok.stages_imp.mean():.2f}", f"{ok.spoof_trait_worst.mean():.2f}"]
            lab = DS[ds] if k == np.inf else ""
            lines.append(f"{lab} & {'none' if k == np.inf else f'{k:.1f}'} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    open(f"{OUT}/tab_cost_body.tex", "w").write("\n".join(lines[:-1]) + "\n")
    MN = {"face_C": "face C", "face_G": "face G", "li_V": "left index", "ri_V": "right index"}
    orders = [f"{DS[ds]}, " + " $>$ ".join(MN[m] for m in e8[e8.dataset == ds].order.iloc[0].split(">")) for ds in DS if (e8.dataset == ds).any()]
    open(f"{OUT}/tab_cost_orders.tex", "w").write("; ".join(orders) + "\n")
    print(open(f"{OUT}/tab_cost_orders.tex").read())
    print(open(f"{OUT}/tab_spoof_body.tex").read()); print(open(f"{OUT}/tab_cost_body.tex").read())

# ------------------------------------------------------------------ IJIS v02: FAR compliance of the deployed proposed designs
if FRESH and os.path.exists(f"{T}/T_fresh_compliance.csv"):
    comp = pd.read_csv(f"{T}/T_fresh_compliance.csv"); lines = []
    def ccell(ds, a, calib, ss, k=3):
        r = comp[(comp.dataset == ds) & np.isclose(comp.alpha, a) & (comp.calib == calib) & (comp.split_set == ss)]
        if len(r) == 0: return ["--"] * k
        r = r.iloc[0]; n = int(r.n); fmt = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
        if calib == "xfit":   # v08: deployed designs (order fixed on fold A), test FAR <= alpha among them, FAR/alpha, FRR
            return [f"{fmt(r.n_deployed)}/{n}", fmt(r.n_met), f"{r.far_over_alpha:.2f}", f"{r.frr_test:.4f}"]
        cnt = fmt(r.far_ok * n)
        return [f"{cnt}/{n}", f"{r.far_over_alpha:.2f}", f"{r.frr_test:.4f}"][:k]
    for ds, lab in [("fing_x_face", "D1"), ("fing_x_fing", "D2"), ("face_x_face", "D3"), ("lfw_x_fing", "D4")]:
        als = sorted(comp[comp.dataset == ds].alpha.unique(), reverse=True)
        ss = "D4 0-9" if ds == "lfw_x_fing" else "fresh"
        for k, a in enumerate(als):
            orig = ["--", "--"] if ds == "lfw_x_fing" else ccell(ds, a, "boot", "original 0-9", 2)
            lines.append(" & ".join([lab if k == 0 else "", AL[a]] + orig + ccell(ds, a, "boot", ss) + ccell(ds, a, "xfit", ss)) + r" \\")
        lines.append(r"\midrule")
    open(f"{OUT}/tab_compliance_body.tex", "w").write("\n".join(lines[:-1]) + "\n")
    print(open(f"{OUT}/tab_compliance_body.tex").read())
