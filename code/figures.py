"""Vector figures (PDF) for the manuscript. IEEE column width 3.5 in, fonts >= 8 pt,
colour-blind-safe validated palette + distinct markers/line styles (readable in B/W)."""
import glob, json, os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import roc, staircase_points, split_subjects, MATCHERS, data_file
from experiment_core import make_blocks

# FIGSTYLE=springer: Springer (IJIS) artwork rules -- 84 mm / 174 mm widths, sans-serif (Arial-metric) lettering
SPRINGER = os.environ.get("FIGSTYLE", "") == "springer"
OUT = "../results/figures_springer" if SPRINGER else "../results/figures"; os.makedirs(OUT, exist_ok=True)
W1, W2 = (84 / 25.4, 174 / 25.4) if SPRINGER else (3.5, 7.16)
NC, HL = (3, 2.2) if SPRINGER else (4, 2.6)   # legend columns / handle length (sans-serif lettering is wider)
plt.rcParams.update({"font.family": "Liberation Sans" if SPRINGER else "STIXGeneral",
                     "mathtext.fontset": "stixsans" if SPRINGER else "stix", "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.4, "pdf.fonttype": 42, "axes.grid": True, "grid.color": "#d9d9d6",
                     "grid.linewidth": 0.4, "axes.edgecolor": "#52514e"})
# validated categorical slots (light mode), fixed order; identity also carried by marker + dash
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STYLE = {  # method -> (label, colour, marker, dash)
    "HYP":          ("Hyperbola (prior GP work)", "#52514e", "x", (0, (1, 1.5))),
    "MONO":         ("Single monomial", C[1], "s", (0, (4, 1.5))),
    "LR-P1-N2":     ("Proposed LR-BB, P1 (N=2)", C[0], "o", "solid"),
    "LR-P1-N2-rel": ("Proposed LR-BB, P1-rel (N=2)", C[2], "^", (0, (6, 1.5, 1, 1.5))),
    "LR-P2-N2":     ("Proposed LR-BB, P2 (N=2)", C[3], "D", (0, (2, 1))),
    "MAXMONO-K3":   ("Max-monomial (K=3)", C[4], "v", (0, (3, 1, 1, 1, 1, 1))),
    "S1-Marcialis": ("Marcialis serial rule", C[5], "P", (0, (5, 2))),
    "P0-Parallel":  ("Parallel sum fusion", "#4a3aa7", "*", (0, (1, 1))),
}

def fig_envelopes(dataset="fing_x_face", matcher="face_C", seed=0, tag="main",
                  methods=("HYP", "MONO", "LR-P1-N2", "LR-P1-N2-rel", "LR-P2-N2")):
    e1 = pd.read_csv(f"../results/E1/{tag}_{dataset}_s{seed}.csv")
    D = dict(np.load(data_file(dataset))); tr, te = make_blocks(D, [matcher], seed); del D
    G = tr["_G"]; S = tr[matcher][0]
    _, far, frr = roc(S[G], S[~G]); x, y = staircase_points(far, frr, "corner")
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.step(x, y, where="pre", color="#0b0b0b", lw=0.9, label="Empirical staircase (train)", zorder=5)   # corners (x_{k+1}, y_k): level y_k ends at x_{k+1}
    xs = np.geomspace(x.min(), x.max(), 400)
    for m in methods:
        r = e1[(e1.matcher == matcher) & (e1.method == m)].iloc[0]; terms = json.loads(r.terms)
        g = sum(a * xs ** b for a, b in terms)
        lab, col, mk, ds = STYLE[m]
        ax.plot(xs, g, color=col, ls=ds, lw=1.3, label=lab, marker=mk, markevery=60, ms=4, mfc="white", mew=0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(max(y[y > 0].min() * 0.5, 1e-3), 1.2)
    ax.set_xlabel("FAR"); ax.set_ylabel("FRR")
    ax.legend(loc="lower left", frameon=False, handlelength=3.0)
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/F2_envelopes_{dataset}_{matcher}_s{seed}.pdf"); plt.close(fig)

def _select(df):
    f = df[df.feasible == True].copy()
    crit = np.where(f.kind == "GP", f.get("frr_pred_cal", f.get("frr_pred_gp")), f.frr_train)
    f["crit"] = crit
    return f.loc[f.groupby(["dataset", "seed", "alpha", "method"])["crit"].idxmin()]

def fig_system(tag="main", methods=("HYP", "MONO", "LR-P1-N2", "LR-P1-N2-rel", "S1-Marcialis", "P0-Parallel")):
    fs = glob.glob(f"../results/E3cal/{tag}_*.csv"); df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    sel = _select(df)
    names = {"fing_x_face": "D1: face+finger (517)", "fing_x_fing": "D2: two fingers (6000)", "face_x_face": "D3: two face matchers (3000)"}
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=False)
    for ax, ds in zip(axs, ["fing_x_face", "fing_x_fing", "face_x_face"]):
        s = sel[sel.dataset == ds]; alphas = sorted(s.alpha.unique(), reverse=True)
        pos = {a: i for i, a in enumerate(alphas)}; off = np.linspace(-0.15, 0.15, len(methods))
        for k, m in enumerate(methods):
            g = s[s.method == m].groupby("alpha")["frr_test"].agg(["mean", "std", "count"]).reset_index()
            if len(g) == 0: continue
            lab, col, mk, dsh = STYLE[m]
            xp = np.array([pos[a] for a in g["alpha"]]) + off[k]; o = np.argsort(xp)
            lo = np.minimum(g["std"].values, g["mean"].values)   # whiskers clipped at 0
            ax.errorbar(xp[o], g["mean"].values[o], yerr=[lo[o], g["std"].values[o]], color=col, ls=dsh, marker=mk, ms=4.5,
                        mfc="white", mew=0.9, capsize=2, lw=1.2, label=lab)
        ax.set_xticks(range(len(alphas))); ax.set_xticklabels([f"$10^{{{int(np.log10(a))}}}$" for a in alphas])
        ax.set_ylim(bottom=0); ax.set_title(names[ds], fontsize=8)
        ax.set_xlabel(r"Target system FAR $\alpha$")
    axs[0].set_ylabel("Test FRR (selected order)")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=3)
    fig.tight_layout(rect=(0, 0.17, 1, 1), pad=0.3); fig.savefig(f"{OUT}/F3_system_frr.pdf"); plt.close(fig)

def fig_conservative(tag="main", method="LR-P1-N2"):
    cal = pd.concat([pd.read_csv(f) for f in glob.glob(f"../results/E3cal/{tag}_*.csv")], ignore_index=True)
    pl = pd.concat([pd.read_csv(f) for f in glob.glob(f"../results/E3/{tag}_*.csv")], ignore_index=True)
    cal = cal[(cal.method == method) & (cal.feasible == True)]; pl = pl[(pl.method == method) & (pl.feasible == True)]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3))
    ax = axs[0]
    for ds, col, mk in [("fing_x_face", C[0], "o"), ("fing_x_fing", C[1], "s"), ("face_x_face", C[2], "^")]:
        d = cal[cal.dataset == ds]
        ax.scatter(d.frr_pred_cal, d.frr_test, s=9, color=col, marker=mk,
                   alpha=0.55, lw=0, label={"fing_x_face": "D1", "fing_x_fing": "D2", "face_x_face": "D3"}[ds])
    lim = [0, 0.45]; ax.plot(lim, lim, color="#0b0b0b", lw=0.8, ls=(0, (4, 2))); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.text(0.30, 0.05, "conservative\n(below the line)", fontsize=8, color="#52514e")
    ax.set_xlabel("Predicted system FRR (envelope-based)")
    ax.set_ylabel("Realized test FRR"); ax.legend(frameon=False, loc="upper left", markerscale=1.5)
    ax = axs[1]
    bins = np.linspace(0, 2.5, 26)
    ax.hist(np.clip(pl.far_test / pl.alpha, 0, 2.5), bins=bins, color=C[1], alpha=0.55, label="Uncalibrated GP design",
            histtype="stepfilled", edgecolor=C[1], hatch="///", lw=0.6)
    ax.hist(np.clip(cal.far_test / cal.alpha, 0, 2.5), bins=bins, color=C[0], alpha=0.55, label="Calibrated deployment",
            histtype="stepfilled", edgecolor=C[0], lw=0.6)
    ax.axvline(1.0, color="#0b0b0b", lw=0.9, ls=(0, (4, 2)))
    ax.set_xlabel(r"Realized test FAR / $\alpha$"); ax.set_ylabel("Number of designs"); ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/F4_conservativeness.pdf"); plt.close(fig)

def fig_scalability(tlim=600):
    """F5: wall-clock time of LR-BB (+OBBT), exact enumeration and DE versus N, |D| and |B| (E7, single core)."""
    fs = sorted(glob.glob("../results/E7/e7_*.csv"))
    if not fs: return
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["complete"] = df["complete"].astype(str).str.lower().eq("true")
    MS = [("LR-BB + OBBT", "LR-BB + OBBT (proposed)", C[0], "o", "solid"),
          ("ENUM", "Exhaustive enumeration", C[1], "s", (0, (4, 1.5))),
          ("DE", "Differential evolution", C[2], "^", (0, (1, 1)))]
    DS = [("fing_x_fing", "li_V", "D2 left index", True), ("fing_x_face", "face_C", "D1 face C", False)]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.35))
    panels = [("N", "N", "Number of terms $N$"), ("D", "nD", r"Fitting points $|\mathcal{D}|$"),
              ("B", "nB", r"Exponent grid size $|\mathcal{B}|$")]
    gaps = {}
    for k, (ax, (w, xcol, xl)) in enumerate(zip(axs, panels)):
        if w == "N": ax.axhline(tlim, color="#52514e", lw=0.7, ls=(0, (3, 2)), zorder=1)
        for ds, mt, dlab, filled in DS:
            g = df[(df.which == w) & (df.dataset == ds) & (df.matcher == mt)]
            for meth, lab, col, mk, dsh in MS:
                h = g[g.method == meth].sort_values(xcol)
                if len(h) == 0: continue
                ok = h[h.complete]; bad = h[~h.complete]
                ax.plot(ok[xcol], ok["time"], color=col, marker=mk, ls=dsh, ms=4, lw=1.1, mew=0.9,
                        mfc=col if filled else "white", zorder=3)
                if meth == "LR-BB + OBBT" and len(bad):   # stopped at time limit with a valid lower bound
                    ax.plot(bad[xcol], bad["time"], color=col, marker=mk, ls="none", ms=4, mew=0.9,
                            mfc=col if filled else "white", zorder=4)
                    if len(ok): ax.plot([ok[xcol].iloc[-1], bad[xcol].iloc[0]], [ok["time"].iloc[-1], bad["time"].iloc[0]],
                                        color=col, ls=dsh, lw=1.1, zorder=3)
                    for _, r in bad.iterrows():
                        gaps.setdefault(int(r.N), []).append(f"{100 * r.cert_gap:.0f}% ({dlab.split()[0]})")
                if meth == "ENUM" and len(bad):             # extrapolated from the enumerated fraction
                    ax.plot(bad[xcol], bad["enum_time_extrapolated"], color=col, marker="x", ls="none", ms=5, mew=1.1, zorder=4)
                    if len(ok): ax.plot([ok[xcol].iloc[-1], bad[xcol].iloc[0]], [ok["time"].iloc[-1], bad["enum_time_extrapolated"].iloc[0]],
                                        color=col, ls=(0, (1, 2)), lw=0.8, zorder=2)
        ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel("Time (s)")
        if w == "N": ax.set_xticks([1, 2, 3, 4, 5])
        else:
            ax.set_xscale("log"); tk = [30, 100, 300, 600] if w == "D" else [31, 61, 151, 301, 601]
            ax.set_xticks(tk); ax.set_xticklabels([str(t) for t in tk]); ax.minorticks_off()
        ax.set_title(f"({'abc'[k]})", fontsize=8, loc="left")
        if w == "N" and gaps:
            txt = "LR-BB gap at time limit:\n" + "\n".join(f"N={n}: " + ", ".join(sorted(v)) for n, v in sorted(gaps.items()))
            ax.set_ylim(top=1e8)
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#0b0b0b")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=col, marker=mk, ls=dsh, ms=4, lw=1.1, label=lab) for _, lab, col, mk, dsh in MS]
    hs += [Line2D([], [], color="#52514e", marker="o", ls="none", ms=4, mfc="#52514e", label="D2 left index (filled)"),
           Line2D([], [], color="#52514e", marker="o", ls="none", ms=4, mfc="white", label="D1 face C (hollow)"),
           Line2D([], [], color=C[1], marker="x", ls="none", ms=5, label="Enumeration, extrapolated"),
           Line2D([], [], color="#52514e", ls=(0, (3, 2)), lw=0.7, label=f"Time limit, panel (a) ({tlim} s)")]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=2.6)
    fig.tight_layout(rect=(0, 0.2, 1, 1), pad=0.3); fig.savefig(f"{OUT}/F5_scalability.pdf"); plt.close(fig)


# ============================================================================ revision figures
def fig_system_rev():
    """F3 (revision): test FRR of the selected calibrated designs (tie-averaged) versus alpha."""
    sel = pd.read_csv("../results/tables/T_rev_selected.csv")
    D4 = SPRINGER and os.path.exists("../results/tables/T_fresh_selected.csv")
    if D4:   # IJIS v02: subset D4 (contemporary face matcher), paper calibration
        f4 = pd.read_csv("../results/tables/T_fresh_selected.csv"); f4 = f4[(f4.dataset == "lfw_x_fing") & (f4.calib == "boot")]
        sel = pd.concat([sel, f4], ignore_index=True)
    meths = [("LR-P1-N2", "Proposed LR-BB, P1", C[0], "o", "solid"), ("HYP", "Hyperbola (prior GP work)", "#52514e", "x", (0, (1, 1.5))),
             ("S1-Marcialis", "Marcialis serial rule", C[5], "P", (0, (5, 2))), ("S4-Direct", "Direct empirical search", C[3], "D", (0, (2, 1))),
             ("S3-SPRT", "SPRT (score accumulation)", C[1], "s", (0, (4, 1.5))), ("P1-LLR", "Parallel LLR fusion", "#4a3aa7", "*", (0, (1, 1))),
             ("P0-Parallel", "Parallel sum fusion", C[4], "v", (0, (3, 1, 1, 1)))]
    names = {"fing_x_face": "D1: face+finger (517)", "fing_x_fing": "D2: two fingers (6000)", "face_x_face": "D3: two face matchers (3000)"}
    if D4: names["lfw_x_fing"] = "D4: face (2021)+fingers (1680)"
    fig, axs = plt.subplots(1, len(names), figsize=(W2, 2.75 if SPRINGER else 2.45))
    for ax, ds in zip(axs, names):
        d = sel[sel.dataset == ds]; als = sorted(d.alpha.unique(), reverse=True); xs = np.arange(len(als))
        for k, (m, lab, col, mk, dsh) in enumerate(meths):
            g = d[d.method == m].groupby("alpha").frr_test.agg(["mean", "std", "size"])
            xx = [i + (k - 3) * 0.05 for i, a in enumerate(als) if a in g.index and g.loc[a, "size"] == 10]
            yy = [g.loc[a, "mean"] for a in als if a in g.index and g.loc[a, "size"] == 10]
            ee = [g.loc[a, "std"] for a in als if a in g.index and g.loc[a, "size"] == 10]
            if not xx: continue
            logy = ds == "lfw_x_fing"                     # D4: FRRs span two orders of magnitude -> logarithmic axis
            yerr = [np.minimum(ee, 0.9 * np.array(yy)), ee] if logy else ee
            ax.errorbar(xx, yy, yerr=yerr, color=col, marker=mk, ls=dsh, ms=4, lw=1.1, capsize=2, mfc="white", mew=0.9, label=lab)
        ax.set_xticks(xs); ax.set_xticklabels([f"$10^{{{int(np.log10(a))}}}$" for a in als])
        ax.set_xlabel(r"Target system FAR $\alpha$")
        if ds == "lfw_x_fing": ax.set_yscale("log")
        else: ax.set_ylim(bottom=0)
        if SPRINGER: ax.set_title(f"({'abcd'[list(names).index(ds)]}) {names[ds].split(':')[0]}", fontsize=8, loc="left")
        else: ax.set_title(names[ds], fontsize=8)
    axs[0].set_ylabel("Test FRR (selected design)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=NC, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=HL)
    fig.tight_layout(rect=(0, 0.24 if SPRINGER else 0.17, 1, 1), pad=0.3); fig.savefig(f"{OUT}/F3_system_frr_rev.pdf"); plt.close(fig)

def fig_conservative_rev(method="LR-P1-N2"):
    """F4 (revision): left, predicted vs realized FRR (bootstrap-calibrated designs; selected designs marked);
    right, test FAR / alpha without calibration, with Clopper-Pearson calibration and with the subject bootstrap."""
    b = pd.concat([pd.read_csv(f) for f in glob.glob("../results/E3b/main_*.csv")], ignore_index=True)
    c = pd.concat([pd.read_csv(f) for f in glob.glob("../results/E3cal/main_*.csv")], ignore_index=True)
    p = pd.concat([pd.read_csv(f) for f in glob.glob("../results/E3/main_*.csv")], ignore_index=True)
    b, c, p = [x[(x.method == method) & (x.feasible == True)] for x in (b, c, p)]
    sel = pd.read_csv("../results/tables/T_rev_selected.csv"); sel = sel[sel.method == method]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.4))
    ax = axs[0]
    for ds, col, mk, lab in [("fing_x_face", C[0], "o", "D1"), ("fing_x_fing", C[1], "s", "D2"), ("face_x_face", C[2], "^", "D3")]:
        d = b[b.dataset == ds]; ax.scatter(d.frr_pred_cal, d.frr_test, s=8, color=col, marker=mk, alpha=0.45, lw=0, label=lab)
        keys = set()
        for _, r in sel[sel.dataset == ds].iterrows():
            for o in str(r.orders).split("|"): keys.add((r.seed, r.alpha, o))
        ds_sel = d[[(s_, a_, o_) in keys for s_, a_, o_ in zip(d.seed, d.alpha, d.order)]]
        ax.scatter(ds_sel.frr_pred_cal, ds_sel.frr_test, s=18, facecolor="none", edgecolor="#0b0b0b", marker=mk, lw=0.7)
    ax.scatter([], [], s=18, facecolor="none", edgecolor="#0b0b0b", marker="o", lw=0.7, label="selected designs")
    lim = [0, 0.45]; ax.plot(lim, lim, color="#0b0b0b", lw=0.8, ls=(0, (4, 2))); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Predicted system FRR (envelope-based)"); ax.set_ylabel("Realized test FRR")
    ax.legend(frameon=False, loc="upper left", markerscale=1.3)
    if SPRINGER:
        for k_, a_ in enumerate(axs): a_.set_title(f"({'ab'[k_]})", fontsize=8, loc="left")
    ax = axs[1]; bins = np.linspace(0, 2.5, 26)
    for x, col, lab, dsh in [(p, C[1], "Uncalibrated", "solid"), (c, "#52514e", "Clopper-Pearson calibration", (0, (4, 1.5))),
                             (b, C[0], "Subject-bootstrap calibration", (0, (1, 1)))]:
        ax.hist(np.clip(x.far_test / x.alpha, 0, 2.5), bins=bins, histtype="step", color=col, lw=1.3, ls=dsh, label=lab,
                weights=np.full(len(x), 1.0 / len(x)))
    ax.axvline(1.0, color="#0b0b0b", lw=0.9, ls=(0, (4, 2)))
    ax.set_xlabel(r"Realized test FAR / $\alpha$"); ax.set_ylabel("Share of designs"); ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/F4_conservativeness_rev.pdf"); plt.close(fig)

def fig_scalability_rev(tlim=600):
    """F5 (revision): time of the exact-dual LR-BB (P1, P2) and the P2 MILP, with the subgradient LR-BB and
    enumeration (P1) from the original runs, versus N, |D| and |B|."""
    new = pd.concat([pd.read_csv(f) for f in glob.glob("../results/E7v2/e7v2_*.csv")], ignore_index=True)
    old = pd.concat([pd.read_csv(f) for f in glob.glob("../results/E7/e7_*.csv")], ignore_index=True)
    for x in (new, old): x["complete"] = x["complete"].astype(str).str.lower().eq("true")
    series = [(new[(new.method == "LR-BB exact dual") & (new.obj == "P1")], "Exact-dual LR-BB, P1", C[0], "o", "solid"),
              (new[(new.method == "LR-BB exact dual") & (new.obj == "P2")], "Exact-dual LR-BB, P2", C[2], "^", (0, (6, 1.5, 1, 1.5))),
              (new[new.method == "MILP (HiGHS)"], "MILP (HiGHS), P2", C[3], "D", (0, (2, 1))),
              (old[old.method == "LR-BB + OBBT"], f"Subgradient LR-BB, P1 ({'preliminary' if SPRINGER else 'first version'})", C[4], "v", (0, (3, 1, 1, 1))),
              (old[old.method == "ENUM"], f"Enumeration, P1 ({'preliminary' if SPRINGER else 'first version'})", C[1], "s", (0, (4, 1.5)))]
    DSs = [("fing_x_fing", "li_V", True), ("fing_x_face", "face_C", False)]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.85 if SPRINGER else 2.4))
    panels = [("N", "N", "Number of terms $N$"), ("D", "nD", r"Fitting points $|\mathcal{D}|$"), ("B", "nB", r"Exponent grid size $|\mathcal{B}|$")]
    for k, (ax, (w, xcol, xl)) in enumerate(zip(axs, panels)):
        if w == "N": ax.axhline(tlim, color="#52514e", lw=0.7, ls=(0, (3, 2)))
        for ds, mt, filled in DSs:
            for df, lab, col, mk, dsh in series:
                h = df[(df.which == w) & (df.dataset == ds) & (df.matcher == mt)].sort_values(xcol)
                if len(h) == 0: continue
                ok = h[h.complete]; bad = h[~h.complete]
                ax.plot(ok[xcol], ok["time"], color=col, marker=mk, ls=dsh, ms=4, lw=1.1, mew=0.9, mfc=col if filled else "white")
                if len(bad): ax.plot(bad[xcol], bad["time"], color=col, marker=mk, ls="none", ms=4, mew=0.9, mfc=col if filled else "white")
        ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel("Time (s)"); ax.set_title(f"({'abc'[k]})", fontsize=8, loc="left")
        if w == "N": ax.set_xticks([1, 2, 3, 4, 5])
        else:
            ax.set_xscale("log"); tk = [30, 100, 300, 600] if w == "D" else [31, 61, 151, 301, 601]
            ax.set_xticks(tk); ax.set_xticklabels([str(t) for t in tk]); ax.minorticks_off()
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=col, marker=mk, ls=dsh, ms=4, lw=1.1, label=lab) for _, lab, col, mk, dsh in series]
    hs += [Line2D([], [], color="#52514e", marker="o", ls="none", ms=4, mfc="#52514e", label="D2 left index (filled)"),
           Line2D([], [], color="#52514e", marker="o", ls="none", ms=4, mfc="white", label="D1 face C (hollow)"),
           Line2D([], [], color="#52514e", ls=(0, (3, 2)), lw=0.7, label=f"Time limit ({tlim} s)")]
    fig.legend(handles=hs, loc="lower center", ncol=NC, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=HL)
    fig.tight_layout(rect=(0, 0.27 if SPRINGER else 0.2, 1, 1), pad=0.3); fig.savefig(f"{OUT}/F5_scalability_rev.pdf"); plt.close(fig)


if __name__ == "__main__":
    # "all" regenerates the figures of the submitted paper (F2 and the revision versions of F3-F5);
    # "old" regenerates the first-version F3-F5 (pair-level calibration), kept for the record.
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("F2", "all"):
        fig_envelopes("fing_x_face", "face_C", 0)
        fig_envelopes("fing_x_fing", "li_V", 0)
    if which in ("F3", "all"): fig_system_rev()
    if which in ("F4", "all"): fig_conservative_rev()
    if which in ("F5", "all"): fig_scalability_rev()
    if which == "old":
        fig_system(); fig_conservative(); fig_scalability()
