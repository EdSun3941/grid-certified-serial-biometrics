"""Stage-5 check: recompute every number quoted in the manuscript from the released CSV files
and compare it with the value printed in the text (at the printed precision).
Usage: python verify_numbers.py   (prints OK/FAIL per item; exit code 1 if any item fails)"""
import glob, sys, numpy as np, pandas as pd

R = "../results"; T = f"{R}/tables"
tab = lambda n: pd.read_csv(f"{T}/{n}.csv")
fails = []

def check(label, value, printed, tol=None):
    """printed: the number as written in the paper; value is rounded to the same number of decimals."""
    if isinstance(printed, str):
        dec = len(printed.split(".")[1]) if "." in printed else 0
        p = float(printed); v = round(float(value), dec)
        ok = abs(v - p) <= (tol if tol is not None else 0.5 * 10 ** (-dec) + 1e-12)
    else:
        ok = value == printed; p = printed; v = value
    print(f"{'OK  ' if ok else 'FAIL'} {label}: computed {value!r} vs printed {printed!r}")
    if not ok: fails.append(label)

DS = {"D1": "fing_x_face", "D2": "fing_x_fing", "D3": "face_x_face"}
# ---------------------------------------------------------------- solver (E2)
e2 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2/e2_*.csv")], ignore_index=True)
bb = e2[e2.method == "LR-BB + OBBT"]
check("LR-BB instances", len(bb), 80); check("LR-BB max gap", bb.cert_gap.max(), "0"); check("LR-BB max excess", bb.rel_to_exact.max(), "0")
tm = e2.groupby(["dataset", "obj", "method"]).time.mean().unstack("method")
sp = tm["ENUM"] / tm["LR-BB + OBBT"]
check("speed-up P1 min", sp.xs("P1", level="obj").min(), "6.2"); check("speed-up P1 max", sp.xs("P1", level="obj").max(), "6.6")
check("speed-up P2 min", sp.xs("P2", level="obj").min(), "8.9"); check("speed-up P2 max", sp.xs("P2", level="obj").max(), "10.2")
g = e2.groupby(["dataset", "obj", "method"]).cert_gap.mean().unstack("method")
check("root gap mean min %", 100 * g["LR-root (no OBBT)"].min(), "51"); check("root gap mean max %", 100 * g["LR-root (no OBBT)"].max(), "85")
check("root gap max %", 100 * e2[e2.method == "LR-root (no OBBT)"].cert_gap.max(), "100")
check("root+OBBT gap mean min %", 100 * g["LR-root + OBBT"].min(), "19"); check("root+OBBT gap mean max %", 100 * g["LR-root + OBBT"].max(), "31")
check("root+OBBT gap max %", 100 * e2[e2.method == "LR-root + OBBT"].cert_gap.max(), "72")
check("root primal worst %", 100 * e2[e2.method == "LR-root (no OBBT)"].rel_to_exact.max(), "28.3")
check("root+OBBT primal worst %", 100 * e2[e2.method == "LR-root + OBBT"].rel_to_exact.max(), "37.6")
rel = e2.groupby(["dataset", "obj", "method"]).rel_to_exact.mean().unstack("method")
check("DE mean best %", -100 * rel["DE"].min(), "0.78"); check("DE mean least %", -100 * rel["DE"].max(), "0.02")
check("DE worst %", 100 * e2[e2.method == "DE"].rel_to_exact.max(), "2.8")
check("NLS mean min %", 100 * rel["NLS-shift"].min(), "20"); check("NLS mean max %", 100 * rel["NLS-shift"].max(), "122")
check("NLS max %", 100 * e2[e2.method == "NLS-shift"].rel_to_exact.max(), "210")
nodes = e2.groupby(["dataset", "obj", "method"]).nodes.mean().unstack("method")
red = 1 - nodes["LR-BB + OBBT"] / nodes["LR-BB (no OBBT)"]
check("OBBT node reduction min %", 100 * red.min(), "1"); check("OBBT node reduction max %", 100 * red.max(), "35")
for m, lo, hi in [("ENUM", "4.9", "6.7"), ("LR-BB (no OBBT)", "0.82", "1.68"), ("LR-BB + OBBT", "0.79", "1.02"), ("DE", "0.58", "0.62"), ("NLS-shift", "0.15", "0.17")]:
    t = tm[m].xs("P1", level="obj"); check(f"time P1 {m} min", t.min(), lo); check(f"time P1 {m} max", t.max(), hi)
for m, lo, hi in [("ENUM", "19.7", "24.7"), ("LR-BB (no OBBT)", "2.87", "4.84"), ("LR-BB + OBBT", "1.94", "2.79"), ("DE", "0.67", "0.70")]:
    t = tm[m].xs("P2", level="obj"); check(f"time P2 {m} min", t.min(), lo); check(f"time P2 {m} max", t.max(), hi)
# ---------------------------------------------------------------- scalability (E7)
s7 = tab("T_scalability")
n3 = s7[(s7.which == "N") & (s7.setting == 3)]
check("N3 LR min", n3.lr_time.min(), "26.9"); check("N3 LR max", n3.lr_time.max(), "99.7")
check("N3 enum min", n3.enum_time.min(), "258.5"); check("N3 enum max", n3.enum_time.max(), "393.7")
check("N3 speed-up min", n3.speedup.min(), "3.9"); check("N3 speed-up max", n3.speedup.max(), "9.6")
n45 = s7[(s7.which == "N") & (s7.setting >= 4)]
check("N>=4 gap min %", 100 * n45.lr_cert_gap.min(), "6"); check("N>=4 gap max %", 100 * n45.lr_cert_gap.max(), "58")
n4 = s7[(s7.which == "N") & (s7.setting == 4)]; n5 = s7[(s7.which == "N") & (s7.setting == 5)]
check("N4 enum frac min %", 100 * n4.enum_fraction.min(), "5"); check("N4 enum frac max %", 100 * n4.enum_fraction.max(), "7")
check("N5 enum frac %", 100 * n5.enum_fraction.max(), "0.2")
check("N4 extrap h min", n4.enum_time.min() / 3600, "2.4"); check("N4 extrap h max", n4.enum_time.max() / 3600, "3.2")
check("N5 extrap h min", n5.enum_time.min() / 3600, "72"); check("N5 extrap h max", n5.enum_time.max() / 3600, "101")
b601 = s7[(s7.which == "B") & (s7.setting == 601)]
check("B601 speed-up D1", b601[b601.dataset == "fing_x_face"].speedup.iloc[0], "7.9"); check("B601 speed-up D2", b601[b601.dataset == "fing_x_fing"].speedup.iloc[0], "25.8")
b151 = s7[(s7.which == "B") & (s7.setting == 151)]
check("grid-DE 0.02 min %", 100 * b151.grid_vs_de.min(), "0.14"); check("grid-DE 0.02 max %", 100 * b151.grid_vs_de.max(), "0.19")
check("grid-DE 0.005 min %", 100 * b601.grid_vs_de.min(), "0.01"); check("grid-DE 0.005 max %", 100 * b601.grid_vs_de.max(), "0.03")
n1 = s7[(s7.which == "N") & (s7.setting == 1) & (s7.dataset == "fing_x_face")]
check("N1 grid vs continuous D1 face C %", 100 * n1.grid_vs_de.iloc[0], "6.6")
raw = tab("T_scalability_raw"); lr = raw[(raw.which == "N") & (raw.method == "LR-BB + OBBT")]
for ds in ["fing_x_face", "fing_x_fing"]:
    v = lr[(lr.dataset == ds) & (lr.N >= 2)].value
    check(f"best value identical N=2..5 {ds}", float(v.max() - v.min()) < 1e-9, True)
# ---------------------------------------------------------------- fitting (E1)
fit = tab("T_fit").set_index(["dataset", "method"])
for k, p in [("D1", "8.0e5"), ("D2", "2.2e11"), ("D3", "2.7e6")]:
    v = fit.loc[(DS[k], "HYP"), "sse_ratio_gm"]; e = int(p.split("e")[1]); check(f"HYP SSE ratio {k}", v / 10 ** e, p.split("e")[0])
mono = [fit.loc[(d, "MONO"), "sse_ratio_gm"] for d in DS.values()]; check("MONO ratio min", min(mono), "1.00"); check("MONO ratio max", max(mono), "1.03")
de = [fit.loc[(d, "DE-N2"), "sse_ratio_gm"] for d in DS.values()]; check("DE ratio min", min(de), "0.99"); check("DE ratio max", max(de), "0.99")
nls = [fit.loc[(d, "NLS-N2"), "sse_ratio_gm"] for d in DS.values()]; check("NLS ratio min", min(nls), "0.84"); check("NLS ratio max", max(nls), "1.17")
mm = [fit.loc[(d, "MAXMONO-K3"), "sse_ratio_gm"] for d in DS.values()]; check("MAXMONO ratio min", min(mm), "4.3"); check("MAXMONO ratio max", max(mm), "1058")
for k, p in [("D1", "15.3"), ("D2", "15.1"), ("D3", "1.2")]:
    check(f"test violation {k} %", 100 * fit.loc[(DS[k], "LR-P1-N2"), "test_viol"], p)
check("all envelopes dominate", bool((fit.dominates == 1).all()), True)
# ---------------------------------------------------------------- system (E3cal)
sc = tab("T_system_cal"); ss = tab("T_system_cal_stats")
get = lambda k, a, m, c="frr_test": sc[(sc.dataset == DS[k]) & (np.isclose(sc.alpha, a)) & (sc.method == m)][c].iloc[0]
pst = lambda k, a, m, c="p_sel_holm": ss[(ss.dataset == DS[k]) & (np.isclose(ss.alpha, a)) & (ss.method == m)][c].iloc[0]
table4 = {("D1", 1e-2): ["0.0066", "0.0421", "0.0066", "0.0062", "0.0143", "0.0162", "0.0124", "0.0031"],
          ("D1", 1e-3): ["0.0100", None, "0.0112", "0.0100", "0.0185", "0.0181", "0.0228", "0.0193"],
          ("D2", 1e-2): ["0.0440", "0.0771", "0.0432", "0.0438", "0.0676", "0.0426", "0.0424", "0.0391"],
          ("D2", 1e-3): ["0.0684", None, "0.0686", "0.0683", "0.0939", "0.0939", "0.0678", "0.0618"],
          ("D2", 1e-4): ["0.1029", None, "0.1028", "0.1028", "0.1225", "0.1225", "0.0989", "0.0899"],
          ("D3", 1e-2): ["0.0751", "0.0972", "0.0753", "0.0754", "0.0957", "0.0849", "0.0761", "0.0837"],
          ("D3", 1e-3): ["0.1614", None, "0.1611", "0.1613", "0.1828", "0.1828", "0.1583", "0.1514"],
          ("D3", 1e-4): ["0.2717", None, "0.2718", "0.2718", "0.2863", "0.2863", "0.2483", "0.2402"]}
cols = ["LR-P1-N2", "HYP", "MONO", "DE-N2", "S1-Marcialis", "S2-Symmetric", "S4-Direct", "P0-Parallel"]
for (k, a), vals in table4.items():
    for m, p in zip(cols, vals):
        r = sc[(sc.dataset == DS[k]) & (np.isclose(sc.alpha, a)) & (sc.method == m)]
        if p is None:
            check(f"T4 {k} {a} {m} infeasible", len(r) == 0 or r.feas_rate.iloc[0] == 0 or np.isnan(r.frr_test.iloc[0]), True)
        else:
            check(f"T4 {k} {a} {m}", r.frr_test.iloc[0], p)
for k, p in [("D1", "0.0041"), ("D2", "0.0021"), ("D3", "0.0018")]: check(f"T4 sd {k} 1e-2", get(k, 1e-2, "LR-P1-N2", "frr_test_sd"), p)
for k, p in [("D1", "33"), ("D2", "30"), ("D3", "88")]: check(f"HYP feasible {k} %", 100 * get(k, 1e-2, "HYP", "feas_rate"), p)
ratios = [get(k, 1e-2, "HYP") / get(k, 1e-2, "LR-P1-N2") for k in DS]; check("HYP FRR ratio min", min(ratios), "1.3"); check("HYP FRR ratio max", max(ratios), "6.4")
for k, p in [("D1", "46"), ("D2", "27"), ("D3", "12")]:
    check(f"reduction vs Marcialis {k} %", 100 * (1 - get(k, 1e-3, "LR-P1-N2") / get(k, 1e-3, "S1-Marcialis")), p)
for k, p in [("D1", "0.125"), ("D2", "0.025"), ("D3", "0.025")]: check(f"p Marcialis {k} 1e-3", pst(k, 1e-3, "S1-Marcialis"), p)
mp = ss[(ss.method == "S1-Marcialis") & ~((ss.dataset == "fing_x_face") & np.isclose(ss.alpha, 1e-3))].p_sel_holm
check("max p vs Marcialis (others)", mp.max(), "0.047"); check("Marcialis always worse", bool((ss[ss.method == "S1-Marcialis"].mean_diff_sel < 0).all()), True)
check("matched-order median diff D1", pst("D1", 1e-3, "S1-Marcialis", "median_diff_all"), "-0.0039")
check("matched-order pairs D1", int(pst("D1", 1e-3, "S1-Marcialis", "n_all")), 598)
check("matched-order rank-biserial D1", pst("D1", 1e-3, "S1-Marcialis", "rank_biserial_all"), "-0.53")
check("symmetric D2 1e-2 p", pst("D2", 1e-2, "S2-Symmetric"), "0.043")
sym = ss[ss.method == "S2-Symmetric"]; check("symmetric worse except one", int((sym.mean_diff_sel > 0).sum()), 1)
fam = ss[ss.method.isin(["MONO", "DE-N2", "NLS-N2", "LR-P1-N3", "LR-P2-N2"])].p_sel_holm.dropna()
check("envelope families min p", fam.min(), "0.16")
check("rel variant D2 1e-3 p", pst("D2", 1e-3, "LR-P1-N2-rel"), "0.025"); check("rel variant D2 1e-4 p", pst("D2", 1e-4, "LR-P1-N2-rel"), "0.025")
check("rel variant D3 1e-4", get("D3", 1e-4, "LR-P1-N2-rel"), "0.2648"); check("rel variant D3 1e-4 p", pst("D3", 1e-4, "LR-P1-N2-rel"), "0.025")
check("maxmono D1 1e-3", get("D1", 1e-3, "MAXMONO-K3"), "0.0510"); check("maxmono D1 1e-3 p", pst("D1", 1e-3, "MAXMONO-K3"), "0.021")
check("parallel D1 1e-3 p", pst("D1", 1e-3, "P0-Parallel"), "0.035"); check("direct D1 1e-3 p", pst("D1", 1e-3, "S4-Direct"), "0.021")
for m, p in [("LR-P1-N2", "79"), ("MONO", "79"), ("DE-N2", "79"), ("S1-Marcialis", "71"), ("S2-Symmetric", "70"), ("S4-Direct", "65"), ("P0-Parallel", "64")]:
    check(f"FAR compliance {m} %", 100 * sc[sc.method == m].far_ok_test_sel.mean(), p)
check("direct D3 compliance %", 100 * sc[(sc.method == "S4-Direct") & (sc.dataset == "face_x_face")].far_ok_test_sel.mean(), "23")
g = sc[sc.method == "LR-P1-N2"].stages_gen; check("stages gen min", g.min(), "1.10"); check("stages gen max", g.max(), "1.31")
# ---------------------------------------------------------------- conservativeness
cv = tab("T_conservativeness")
cget = lambda reg, k, a, ch, c: cv[(cv.regime == reg) & (cv.dataset == DS[k]) & np.isclose(cv.alpha, a) & (cv.chain == ch) & (cv.method == "LR-P1-N2")][c].iloc[0]
check("D1 1e-3 plain test", 100 * cget("plain", "D1", 1e-3, "all", "far_ok_test"), "37.2"); check("D1 1e-3 cal test", 100 * cget("calibrated", "D1", 1e-3, "all", "far_ok_test"), "82.3")
check("D1 1e-3 plain train", 100 * cget("plain", "D1", 1e-3, "all", "far_ok_train"), "57.5"); check("D1 1e-3 cal train", 100 * cget("calibrated", "D1", 1e-3, "all", "far_ok_train"), "100")
for a, pc, pp in [(1e-4, "100", "97.5"), (1e-3, "100", "100"), (1e-2, "92.5", "90")]:
    check(f"D2 {a} cal", 100 * cget("calibrated", "D2", a, "all", "far_ok_test"), pc); check(f"D2 {a} plain", 100 * cget("plain", "D2", a, "all", "far_ok_test"), pp)
check("D3 corr ratio plain", cget("plain", "D3", 1e-4, "correlated-pair", "far_ratio_median"), "3.14"); check("D3 corr ratio cal", cget("calibrated", "D3", 1e-4, "correlated-pair", "far_ratio_median"), "1.02")
check("D3 corr cal compliance %", 100 * cget("calibrated", "D3", 1e-4, "correlated-pair", "far_ok_test"), "50")
check("D3 single cal compliance %", 100 * cget("calibrated", "D3", 1e-4, "single", "far_ok_test"), "40")
check("FRR validity D1 1e-3 %", 100 * cget("calibrated", "D1", 1e-3, "all", "frr_ok_test"), "93"); check("FRR validity D2 1e-3 %", 100 * cget("calibrated", "D2", 1e-3, "all", "frr_ok_test"), "100")
for a in [1e-3, 1e-4]: check(f"FRR validity D3 corr {a} %", 100 * cget("calibrated", "D3", a, "correlated-pair", "frr_ok_test"), "0")
dec = tab("T_decomposition").set_index(["dataset", "alpha"])
for k, vals in [("D1", ["0.073", "0.067", "0.006", "0.012", "0.015"]), ("D2", ["0.134", "0.126", "0.072", "0.093", "0.095"]), ("D3", ["0.182", "0.171", "0.120", "0.173", "0.170"])]:
    row = dec.loc[(DS[k], 1e-3)]
    for c, p in zip(["frr_pred_gp", "frr_pred_exact", "frr_emp_model", "frr_train", "frr_test"], vals): check(f"decomp {k} {c}", row[c], p)
r = dec.loc[("fing_x_fing", 1e-3)]
for lab, v, p in [("series", r.frr_pred_exact / r.frr_pred_gp - 1, "-0.06"), ("envelope", r.frr_emp_model / r.frr_pred_exact - 1, "-0.43"),
                  ("dependence", r.frr_train / r.frr_emp_model - 1, "0.30"), ("generalization", r.frr_test / r.frr_train - 1, "0.02")]:
    check(f"decomp D2 {lab}", v, p)
p1 = tab("T_prop1_check")
cor = p1[p1.dominance.str.startswith("corner")]; check("Prop1 corner designs", int(cor.n_designs.sum()), 2044); check("Prop1 corner all ok", bool((cor.frac_realized_le_pred == 1).all()), True)
smp = p1[(p1.dominance.str.startswith("sample")) & (p1.method == "LR-P1-N2")]
check("Prop1 sample frac %", 100 * smp.frac_realized_le_pred.iloc[0], "88.75"); check("Prop1 sample max excess", smp.max_excess.iloc[0], "0.0055")
nz = tab("T_noise"); r = nz[(nz.method == "LR-P1-N2") & np.isclose(nz.alpha, 1e-2)].iloc[0]
check("noise FAR before", r.far_test, "0.0096"); check("noise FAR after", r.far_test_n10, "0.0113"); check("noise FRR unchanged", abs(r.frr_test_n10 - r.frr_test) < 0.001, True)
# ---------------------------------------------------------------- ablations
ab = tab("T_ablation_paired"); aget = lambda v, m, a, c: ab[(ab.variant == v) & (ab.method == m) & np.isclose(ab.alpha, a)][c].iloc[0]
check("abl sample variant", aget("abl_sample", "LR-P1-N2", 1e-3, "frr_test_variant"), "0.0097"); check("abl sample main", aget("abl_sample", "LR-P1-N2", 1e-3, "frr_test_main"), "0.0093")
check("abl region variant", aget("abl_region", "LR-P1-N2", 1e-3, "frr_test_variant"), "0.0432"); check("abl region p", aget("abl_region", "LR-P1-N2", 1e-3, "p_wilcoxon"), "0.002")
check("abl N4", aget("abl_N", "LR-P1-N4", 1e-3, "frr_test_variant"), "0.0077"); check("abl N5", aget("abl_N", "LR-P1-N5", 1e-3, "frr_test_variant"), "0.0077")
check("abl b001", aget("sens_b001", "LR-P1-N2", 1e-3, "frr_test_variant"), "0.0077"); check("abl b005", aget("sens_b005", "LR-P1-N2", 1e-3, "frr_test_variant"), "0.0077")
check("abl nb60", aget("sens_nb60", "LR-P1-N2", 1e-3, "frr_test_variant"), "0.0085"); check("abl nb240", aget("sens_nb240", "LR-P1-N2", 1e-3, "frr_test_variant"), "0.0062")
sens = tab("T_ablation_sensitivity"); sget = lambda t, a: sens[(sens.tag == t) & (sens.method == "LR-P1-N2") & np.isclose(sens.alpha, a)].frr_test.iloc[0]
check("frac 0.3", sget("frac03", 1e-3), "0.0083"); check("frac 0.7", sget("frac07", 1e-3), "0.0103")
rf = tab("T_reject_far"); r = rf[(rf.dataset == "fing_x_face") & np.isclose(rf.alpha, 1e-3)].iloc[0]
check("reject FAR > 0.1 %", 100 * r.frac_a_rej_gt_0p1, "99.5"); check("reject FAR median", r.a_rej_median, "0.84")
# ---------------------------------------------------------------- datasets (Table II)
dsn = tab("T_datasets").set_index(["dataset", "matcher"])
for (k, m), (eer, frr) in {("D1", "face_C"): ("0.044", "0.161"), ("D1", "face_G"): ("0.058", "0.222"), ("D1", "li_V"): ("0.085", "0.168"),
                           ("D1", "ri_V"): ("0.050", "0.101"), ("D2", "li_V"): ("0.078", "0.188"), ("D2", "ri_V"): ("0.053", "0.121"),
                           ("D3", "face_C"): ("0.053", "0.186"), ("D3", "face_G"): ("0.063", "0.236")}.items():
    check(f"EER {k} {m}", dsn.loc[(DS[k], m), "EER"], eer); check(f"FRR@1e-3 {k} {m}", dsn.loc[(DS[k], m), "FRR@FAR<=0.001"], frr)
cr = tab("T_correlations")
check("corr C-G D1", cr[(cr.dataset == "fing_x_face") & (cr.pair == "face_C-face_G")].genuine_pearson.iloc[0], "0.49")
check("corr L-R D1", cr[(cr.dataset == "fing_x_face") & (cr.pair == "li_V-ri_V")].genuine_pearson.iloc[0], "0.41")
ff = cr[(cr.dataset == "fing_x_face") & cr.pair.str.contains("face") & cr.pair.str.contains("_V")].genuine_pearson
check("corr face-finger min", ff.min(), "-0.12"); check("corr face-finger max", ff.max(), "-0.02")
check("corr L-R D2", cr[cr.dataset == "fing_x_fing"].genuine_pearson.iloc[0], "0.41"); check("corr C-G D3", cr[cr.dataset == "face_x_face"].genuine_pearson.iloc[0], "0.48")
# ---------------------------------------------------------------- D1 split sizes ("about 260 genuine, 6.7e4 impostor per test half")
from data import data_file, split_subjects
Dd = np.load(data_file("fing_x_face")); Gd = Dd["genuine"]
cnt = [(Gd[np.ix_(r, c)].sum(), (~Gd[np.ix_(r, c)]).sum()) for r, c in [split_subjects(Dd["row_subject"], Dd["col_subject"], sd)[2:] for sd in range(10)]]
check("D1 test genuine ~260", all(250 <= g <= 270 for g, _ in cnt), True); check("D1 test impostor ~6.7e4", np.mean([i for _, i in cnt]) / 1e4, "6.7")
check("D1 cross-modal cal compliance 1e-3 %", 100 * cget("calibrated", "D1", 1e-3, "cross-modal", "far_ok_test"), "86")
check("D1 cross-modal cal compliance 1e-2 %", 100 * cget("calibrated", "D1", 1e-2, "cross-modal", "far_ok_test"), "85")
check("D1 cross-modal FRR validity", bool(cget("calibrated", "D1", 1e-3, "cross-modal", "frr_ok_test") == 1 and cget("calibrated", "D1", 1e-2, "cross-modal", "frr_ok_test") == 1), True)
check("D3 calibrated train FAR ok", bool((cv[(cv.regime == "calibrated") & (cv.dataset == "face_x_face") & (cv.method == "LR-P1-N2")].far_ok_train == 1).all()), True)
print(f"\n{len(fails)} failures" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
