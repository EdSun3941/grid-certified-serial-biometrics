"""Stage-5 check of the submitted manuscript: every number quoted in the text is recomputed from the released
result files and compared at the printed precision; each check also confirms that the printed phrase occurs in
the LaTeX source, so the checks stay tied to the current text.  The main-paper table bodies are regenerated and
compared with the files in ../paper/tables.
Usage: python verify_numbers.py      (exit code 1 if any check fails)"""
import glob, hashlib, os, re, subprocess, sys
import numpy as np, pandas as pd

R = "../results"; T = f"{R}/tables"; P = os.environ.get("PAPER", "../paper")   # PAPER=../paper_ijis checks the IJIS version
SPRINGER = "ijis" in P; SUPP = f"{P}/ESM_1.tex" if SPRINGER else f"{P}/supplementary.tex"
tab = lambda n: pd.read_csv(f"{T}/{n}.csv")
SRC = {os.path.basename(f): open(f).read() for f in glob.glob(f"{P}/sections/*.tex") + [SUPP] + ([f"{P}/main.tex"] if SPRINGER else [])}
ALL = "\n".join(SRC.values())
fails = []; n_ok = 0
IJ = lambda tifs, ijis: ijis if SPRINGER else tifs    # phrase of the TIFS or of the IJIS text

def _norm(s): return re.sub(r"\s+", " ", s)
def check(label, value, printed, phrase=None, scale=1.0, tol=None):
    """value (times scale) rounded to the decimals of `printed` must equal it; `phrase` must occur in the source."""
    global n_ok
    dec = len(printed.split(".")[1]) if "." in printed else 0
    v = round(float(value) * scale, dec); p = float(printed)
    ok = (abs(float(value) * scale - p) <= tol) if tol is not None else (abs(v - p) <= 1e-12 or v == p)
    txt_ok = True if phrase is None else _norm(phrase) in _norm(ALL)
    status = "OK  " if (ok and txt_ok) else "FAIL"
    print(f"{status} {label}: computed {float(value) * scale:.6g} -> {v} vs printed {printed}" + ("" if txt_ok else f"   [phrase not found: {phrase!r}]"))
    if ok and txt_ok: n_ok += 1
    else: fails.append(label)
def check_true(label, cond, phrase=None):
    global n_ok
    txt_ok = True if phrase is None else _norm(phrase) in _norm(ALL)
    ok = bool(cond) and txt_ok
    print(f"{'OK  ' if ok else 'FAIL'} {label}" + ("" if txt_ok else f"   [phrase not found: {phrase!r}]"))
    if ok: n_ok += 1
    else: fails.append(label)

DS = {"D1": "fing_x_face", "D2": "fing_x_fing", "D3": "face_x_face"}
print("=" * 30, "Data (Section V, Table I)")
dst = tab("T_datasets"); cor = tab("T_correlations")
for d, m, eer, f3 in [("D1", "face_C", "0.044", "0.161"), ("D1", "face_G", "0.058", "0.222"), ("D1", "li_V", "0.085", "0.168"),
                      ("D1", "ri_V", "0.050", "0.101"), ("D2", "li_V", "0.078", "0.188"), ("D2", "ri_V", "0.053", "0.121"),
                      ("D3", "face_C", "0.053", "0.186"), ("D3", "face_G", "0.063", "0.236")]:
    r = dst[(dst.dataset == DS[d]) & (dst.matcher == m)].iloc[0]
    check(f"{d} {m} EER", r.EER, eer); check(f"{d} {m} FRR@1e-3", r["FRR@FAR<=0.001"], f3)
check_true("D1 counts 517/266,772", (dst[dst.dataset == DS["D1"]].n_genuine == 517).all() and (dst[dst.dataset == DS["D1"]].n_impostor == 266772).all(), "517 /\\\\266{,}772")
check("D2 impostors 3.6e7", dst[dst.dataset == DS["D2"]].n_impostor.iloc[0] / 1e7, "3.6", "6000 /\\\\$3.6\\times10^{7}$")
check("D3 impostors 1.8e7", dst[dst.dataset == DS["D3"]].n_impostor.iloc[0] / 1e7, "1.8", "6000 /\\\\$1.8\\times10^{7}$")
gc = lambda ds, pr: cor[(cor.dataset == DS[ds]) & (cor.pair == pr)].iloc[0]
check("D1 C-G genuine corr", gc("D1", "face_C-face_G").genuine_pearson, "0.49", "C--G 0.49")
check("D1 L-R genuine corr", gc("D1", "li_V-ri_V").genuine_pearson, "0.41", "L--R 0.41")
ff = cor[(cor.dataset == DS["D1"]) & cor.pair.str.contains("face") & cor.pair.str.contains("_V")].genuine_pearson
check("D1 face-finger corr min", ff.min(), "-0.12", "$-0.12$ to $-0.02$"); check("D1 face-finger corr max", ff.max(), "-0.02")
check("D2 L-R genuine corr", gc("D2", "li_V-ri_V").genuine_pearson, "0.41")
check("D3 C-G genuine corr", gc("D3", "face_C-face_G").genuine_pearson, "0.48", "C--G 0.48")
check("D3 genuine corr (discussion)", gc("D3", "face_C-face_G").genuine_pearson, "0.48", "genuine-score correlation 0.48")
same = cor[cor.pair.isin(["face_C-face_G", "li_V-ri_V"]) & cor.dataset.isin(list(DS.values()))].impostor_pearson   # BSSR1 subsets D1-D3
check("same-modality impostor corr min", same.min(), "0.12", "impostor-score correlations of the same-modality pairs are 0.12--0.16")
check("same-modality impostor corr max", same.max(), "0.16")
# failure-to-acquire scores (score -1) in the processed data
try:
    sys.path.insert(0, "."); from data import data_file
    D1 = np.load(data_file("fing_x_face")); D3 = np.load(data_file("face_x_face"))
    rows1 = set(); rows1_any = set()
    for k in D1.files:
        if k.startswith("S_"):
            rows1 |= set(np.nonzero((D1[k] == -1).all(axis=1))[0]); rows1_any |= set(np.nonzero((D1[k] == -1).any(axis=1))[0])
    check("D1 probe rows with failure scores", len(rows1_any), "1", "one D1 probe")
    fc = D3["S_face_C"]; g3 = D3["genuine"]
    check("D3 face C failure share %", float((fc == -1).mean()), "0.2", "0.2\\% of the D3 face-C comparisons", scale=100)
    check("D3 face C failed genuine", int(((fc == -1) & g3).sum()), "12", "including 12 genuine ones")
    check_true("no failure scores in D3 face G / D2", not (D3["S_face_G"] == -1).any())
    del D1, D3
except FileNotFoundError as e:
    print("SKIP failure-score checks (processed BSSR1 data not available):", e)
check_true("D1 test genuine about 260 (258 or 259)", abs(517 / 2 - 260) < 2, "about 260 genuine")
check("D1 training genuine", 517 // 2, "258", "hold only 258 genuine comparisons")
check("D1 test impostors 6.7e4", (517 / 2) * (517 / 2 - 1) / 1e4, "6.7", "$6.7\\times10^{4}$ impostor comparisons")
check("D1 FRR resolution", 1 / 259, "0.004", "an FRR resolution of 0.004")
check("D1 ordered subsets", 4 + 12 + 24 + 24, "64", "All 64 ordered subsets")
check("corrected t variance factor", 1 + 10 * 1.0, "11", "=11$ for $K=10$ splits")

print("=" * 30, "Envelope tightness (Section VI-A, introduction)")
fit = tab("T_fit"); fv = lambda ds, m, c: fit[(fit.dataset == DS[ds]) & (fit.method == m)][c].iloc[0]
check("HYP SSE ratio D1 (x1e5)", fv("D1", "HYP", "sse_ratio_gm") / 1e5, "8.0", "$8.0\\times10^{5}$ (D1)")
check("HYP SSE ratio D2 (x1e11)", fv("D2", "HYP", "sse_ratio_gm") / 1e11, "2.2", "$2.2\\times10^{11}$ (D2)")
check("HYP SSE ratio D3 (x1e6)", fv("D3", "HYP", "sse_ratio_gm") / 1e6, "2.7", "$2.7\\times10^{6}$ (D3)")
check("intro HYP ratio low (x1e5)", fit[fit.method == "HYP"].sse_ratio_gm.min() / 1e5, "8", "$8\\times10^{5}$ to $2\\times10^{11}$")
check("intro HYP ratio high (x1e11)", fit[fit.method == "HYP"].sse_ratio_gm.max() / 1e11, "2")
for m, lo, hi, ph in [("MONO", "1.00", "1.03", "single monomial (ratio 1.00--1.03)"), ("NLS-N2", "0.84", "1.17", "scaled least-squares fit (0.84--1.17)"),
                      ("MAXMONO-K3", "4.3", "1058", "looser (4.3--1058)")]:
    r = fit[fit.method == m].sse_ratio_gm; check(f"{m} SSE ratio min", r.min(), lo, ph); check(f"{m} SSE ratio max", r.max(), hi)
check("DE SSE ratio (all)", fit[fit.method == "DE-N2"].sse_ratio_gm.min(), "0.99", "differential evolution (0.99)")
check("DE SSE ratio max", fit[fit.method == "DE-N2"].sse_ratio_gm.max(), "0.99")
for d, pr in [("D1", "15.3"), ("D2", "15.1"), ("D3", "1.2")]:
    check(f"test corners above envelope {d} %", fv(d, "LR-P1-N2", "test_viol"), pr, f"{pr}\\% ({d})" if d != "D3" else "1.2\\% (D3)", scale=100)

print("=" * 30, "Solvers (Sections IV, VI-B), against the independent exact reference")
e2 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2/e2_*.csv")], ignore_index=True)
v2 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2v2/e2v2_*_N2.csv")], ignore_index=True)
ref = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2ref/ref_*.csv")], ignore_index=True)
KEY = ["dataset", "seed", "matcher", "obj"]
e2 = e2.merge(ref[KEY + ["value"]].rename(columns={"value": "opt"}), on=KEY); v2 = v2.merge(ref[KEY + ["value"]].rename(columns={"value": "opt"}), on=KEY)
e2["excess"] = (e2.value - e2.opt) / e2.opt; e2["short"] = (e2.opt - e2.lb) / e2.opt
v2["excess"] = (v2.value - v2.opt) / v2.opt; v2["short"] = (v2.opt - v2.root_lb) / v2.opt
sg = e2[e2.method == "LR-root (no OBBT)"]
check("subgradient root bound shortfall P1 mean %", sg[sg.obj == "P1"].short.mean(), "61", "on average 61\\% (P1) and 71\\% (P2) below the optimum", scale=100)
check("subgradient root bound shortfall P2 mean %", sg[sg.obj == "P2"].short.mean(), "71", scale=100)
check("abstract: subgradient 61%-71%", sg[sg.obj == "P1"].short.mean(), "61", IJ("left the root bound on average 61\\%--71\\% below the optimum", "(earlier subgradient bound: 61\\% below on average)"), scale=100)
check("Sec. IV: subgradient 61% / 71%", sg[sg.obj == "P2"].short.mean(), "71", "left the root bound on average 61\\% (P1) and 71\\% (P2) below the optimum on the BSSR1 fitting sets", scale=100)
check_true("subgradient root bound often zero", (sg.short >= 1 - 1e-9).sum() >= 10, "often at zero")
check("subgradient primal worst %", sg.excess.max(), "28.3", "up to 28.3\\% worse than the optimum", scale=100)
en = e2[e2.method == "ENUM"]; bb1 = e2[e2.method == "LR-BB + OBBT"]
bad = en[en.excess > 1e-6].sort_values("excess")
check("first-version enumeration wrong instances", len(bad), "2", "exceeded the optimum in two P1 instances, by 0.04\\% and 0.16\\%")
check("first-version excess 1 %", bad.excess.iloc[0], "0.04", scale=100); check("first-version excess 2 %", bad.excess.iloc[1], "0.16", scale=100)
check_true("first-version LR-BB wrong in the same instances and claimed zero gap", set(map(tuple, bb1[bb1.excess > 1e-6][KEY].values)) == set(map(tuple, bad[KEY].values)) and (bb1.cert_gap.max() <= 1e-4))
ex = v2[v2.method == "LR-BB exact dual"]; p1 = ex[ex.obj == "P1"]; p2 = ex[ex.obj == "P2"]
check("P1 root bound = optimum", int((p1.short <= 1e-4).sum()), "39", "the root bound of P1 equaled the optimum in 39 of the 40 instances")
check("P1 root shortfall other %", p1.short.max(), "0.26", "fell short by 0.26\\% in the other, which LR-BB closed after 41 nodes", scale=100)
check("P1 nodes of that instance", p1.loc[p1.short.idxmax()].nodes, "41")
check("abstract: root gap 39 of 40", int((p1.short <= 1e-4).sum()), "39", IJ("least-squares problem in 39 of 40 instances, whereas", "the exact dual closed the root gap in 39 of 40 instances (earlier"))
check("intro: root gap 39 of 40", int((p1.short <= 1e-4).sum()), "39", "closed the root gap in 39 of 40 instances and left 0.26\\% in the other")
check("conclusion: root gap 39 of 40", int((p1.short <= 1e-4).sum()), "39", IJ("in 39 of 40 BSSR1 instances, left 0.26\\% in the other", "in 39 of 40 BSSR1 instances, mostly because two adjacent grid exponents can emulate an intermediate one, left 0.26\\% in the other"))
check("P2 root bound = optimum", int((p2.short <= 1e-4).sum()), "37", "For P2 the root bound equaled the optimum in 37 instances (at most 1.86\\% below)")
check("P2 root shortfall max %", p2.short.max(), "1.86", scale=100)
check("P2 mean nodes", p2.nodes.mean(), "3.6", "LR-BB needed 3.6 nodes on average (at most 44)"); check("P2 max nodes", p2.nodes.max(), "44")
check_true("every exact-dual run reproduced the optimum and certified it", ex.complete.all() and ex.excess.abs().max() < 1e-9 and (ex.n_open_leaves == 0).all(),
           "Every run reproduced the reference optimum and certified it within $\\epsilon$")
milp = v2[v2.method == "MILP (HiGHS)"]
check_true("MILP within 1e-4 of optimum", milp.excess.abs().max() <= 1e-4, "reproduced the optimum within its relative tolerance of $10^{-4}$")
de = e2[e2.method == "DE"]
check("DE P1 mean below optimum %", -de[de.obj == "P1"].excess.mean(), "0.5", "returned P1 values 0.5\\% below the grid optimum on average", scale=100)
check_true("DE P1 always below the grid optimum", (de[de.obj == "P1"].excess < 0).all())
check("DE P2 min %", -de[de.obj == "P2"].excess.min(), "1.9", "P2 values between 1.9\\% below and 2.8\\% above it", scale=100)
check("DE P2 max %", de[de.obj == "P2"].excess.max(), "2.8", scale=100)
mc = tab("T_rev_monoc")
adj = lambda sup: len(sup.split()) == 1 or (len(sup.split()) == 2 and abs(float(sup.split()[1]) - float(sup.split()[0])) <= 0.021)
mc["compact"] = mc.support.map(adj)
check("P1 instances with one or two adjacent exponents", int(mc.compact.sum()), "38", "In 38 of the 40 P1 instances, the optimal weights")
d1c = mc[(mc.dataset == DS["D1"]) & (mc.seed == 0) & (mc.matcher == "face_C")].support.iloc[0]
check_true("D1 face C support -0.26 -0.24", d1c == "-0.26 -0.24", IJ("(for D1 face C, $-0.26$ and $-0.24$)", "(for D1 face C, split 0, $-0.26$ and $-0.24$)"))
good = mc[mc.compact]; badm = mc[~mc.compact].sort_values("rel_cont")
check_true("continuous monomial never worse in the compact instances", (good.rel_cont <= 1e-9).all())
check("continuous monomial best improvement %", -good.rel_cont.min(), "1.4", "up to 1.4\\% better than the two-term grid optimum", scale=100)
check("continuous monomial worse 1 %", badm.rel_cont.iloc[0], "6.6", "the single monomial was 6.6\\% and 23.4\\% worse", scale=100)
check("continuous monomial worse 2 %", badm.rel_cont.iloc[1], "23.4", scale=100)
key = lambda d: set(zip(d.dataset, d.seed, d.matcher))
check_true("the gap instance is one of the two non-compact ones", key(p1[p1.short > 1e-4]) <= key(mc[~mc.compact]) and len(key(p1[p1.short > 1e-4])) == 1,
           IJ("one of them was the only instance with a duality gap", "was the only instance with a duality gap"))
check_true("steep/shallow example -1.62 -0.28", "-1.62 -0.28" in set(mc.support), IJ("(for example, $-1.62$ and $-0.28$)", "and $-1.62$ and $-0.28$ for D1 face C, split 3)"))
if SPRINGER:
    gi = badm.iloc[0]; v2g = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2v2/e2v2_*_N2.csv")], ignore_index=True)
    v2g = v2g[(v2g.method == "LR-BB exact dual") & (v2g.obj == "P1")]; gap_ = v2g.loc[((v2g.ref_value - v2g.root_lb) / v2g.ref_value).idxmax()]
    check_true("IJIS: the 6.6% instance is D3 face G with three support exponents and is the duality-gap instance",
               gi.dataset == DS["D3"] and gi.matcher == "face_G" and int(gi.n_support) == 3 and gi.support == "-0.64 -0.18 -0.16"
               and (gap_.dataset, gap_.seed, gap_.matcher) == (gi.dataset, gi.seed, gi.matcher) and int(gi.seed) == 4
               and int(badm.iloc[1].seed) == 3 and badm.iloc[1].matcher == "face_C" and badm.iloc[1].dataset == DS["D1"] and int(mc[mc.support == "-0.26 -0.24"].seed.min()) == 0,
               "($-0.64$ with the adjacent pair $-0.18$ and $-0.16$ for D3 face G, split 4")
check("discussion: continuous monomial up to 1.4%", -good.rel_cont.min(), "1.4", "improved on the two-term grid optimum by up to 1.4\\%", scale=100)
e7 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E7v2/e7v2_*.csv")], ignore_index=True)
nN = e7[(e7.which == "N") & (e7.method == "LR-BB exact dual") & (e7.obj == "P1") & (e7.N >= 2)]
check_true("P1 N=2..5 certified at root", (nN.nodes == 1).all() and nN.complete.all(), "P1 was certified at the root for every $N$ from 2 to 5")
check_true("P1 values identical N=2..5", all(np.ptp(g.value) < 1e-9 * g.value.max() for _, g in nN.groupby("dataset")), IJ("the best values for $N=2$ to $5$ were identical", "the best P1 values for $N=2$ to $5$ were identical"))
t = nN.groupby("dataset").time
check("E7 D2 P1 time min", t.min()[DS["D2"]], "0.46", "0.46--0.62~s on D2"); check("E7 D2 P1 time max", t.max()[DS["D2"]], "0.62")
check("E7 D1 P1 time min", t.min()[DS["D1"]], "0.74", "0.74--1.03~s on D1 face C"); check("E7 D1 P1 time max", t.max()[DS["D1"]], "1.03")
p2r = e7[(e7.obj == "P2")]
check_true("all P2 runs certified", p2r.complete.all())
check_true("LR-BB P2 max time <= 2.1 s", p2r[p2r.method == "LR-BB exact dual"].time.max() <= 2.1, "LR-BB certified every case within 2.1~s and the MILP within 5.2~s")
check_true("MILP max time <= 5.2 s", p2r[p2r.method == "MILP (HiGHS)"].time.max() <= 5.2)
b601 = e7[(e7.which == "B") & (e7.nB == 601) & (e7.obj == "P1") & (e7.method == "LR-BB exact dual")].time
check("P1 time m=601 min", b601.min(), "45", "(45--59~s at $m=601$)"); check("P1 time m=601 max", b601.max(), "59")
s7 = tab("T_scalability"); n45 = s7[(s7.which == "N") & (s7.setting >= 4)]
check_true("subgradient LR-BB and enumeration not certified for N>=4 in 600 s", (~n45.lr_complete).all() and (~n45.enum_complete).all(),
           "could not certify $N\\ge4$ within 600~s")
xd = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E1xd/xd_*.csv")], ignore_index=True); xm = xd[xd.method == "LR-P1-N2"]
check("main envelopes identical after refit", int((xm.max_rel_curve_diff < 1e-6).sum()), "75", "coincided with the exact-dual fits for 75 of the 80 main envelopes")
check("main envelopes: largest difference %", xm.max_rel_curve_diff.max(), "0.75", "differed by at most 0.75\\% otherwise", scale=100)
check_true("all refits certified", xd.complete_new.astype(bool).all())

print("=" * 30, "System-level results (Section VI-C, abstract)")
sy = tab("T_rev_system"); st = tab("T_rev_stats"); pr = st[st.family == "primary"]
val = lambda ds, a, m, c="frr_test": sy[(sy.dataset == DS[ds]) & np.isclose(sy.alpha, a) & (sy.method == m)][c].iloc[0]
stat = lambda ds, a, m: pr[(pr.dataset == DS[ds]) & np.isclose(pr.alpha, a) & (pr.method == m)].iloc[0]
hyp_any = sy[(sy.method == "HYP") & (sy.alpha <= 1e-3)]
check_true("HYP infeasible at alpha<=1e-3", (hyp_any.n_seeds == 0).all() if len(hyp_any) else True, IJ("admitted no feasible design at $\\far\\le10^{-3}$", "yielded no feasible design at $\\alpha\\le10^{-3}$ on D1--D3"))
feas = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E3b/main_*.csv")], ignore_index=True)
check_true("HYP: no feasible design in any order/split at alpha<=1e-3", not feas[(feas.method == "HYP") & (feas.alpha <= 1e-3)].feasible.any())
ratios = [val(d, 1e-2, "HYP") / val(d, 1e-2, "LR-P1-N2") for d in DS]
check("HYP/proposed ratio min", min(ratios), "1.1", "1.1--6.0 times that of the proposed design"); check("HYP/proposed ratio max", max(ratios), "6.0")
check("D2 1e-3 proposed", val("D2", 1e-3, "LR-P1-N2"), "0.0686", "0.0686 against 0.0959")
check("D2 1e-3 Marcialis", val("D2", 1e-3, "S1-Marcialis"), "0.0959")
s = stat("D2", 1e-3, "S1-Marcialis")
check("D2 1e-3 CI low", s.ci_lo, "-0.032", "$[-0.032,-0.022]$"); check("D2 1e-3 CI high", s.ci_hi, "-0.022")
sig = lambda m: [(d, a) for d in ["D2", "D3"] for a in [1e-2, 1e-3, 1e-4] if stat(d, a, m).p_holm < 0.05 and stat(d, a, m).mean_diff < 0]
mar = sig("S1-Marcialis"); sym = sig("S2-Symmetric")
check("Marcialis significant settings (D2,D3)", len(mar), "5", "lower FRR than the Marcialis rule in five of the six D2 and D3 settings")
check("symmetric significant settings (D2,D3)", len(sym), "3", "and than symmetric rejection in three")
b3b = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E3b/main_*.csv")], ignore_index=True)
b3b = b3b[b3b.method.isin(["S1-Marcialis", "S2-Symmetric"]) & (b3b.feasible == True)]
pvt = b3b.pivot_table(index=["dataset", "seed", "alpha", "order"], columns="method", values="thr", aggfunc="first").dropna().reset_index()
same = {(d, a): bool((g["S1-Marcialis"] == g["S2-Symmetric"]).all()) for (d, a), g in pvt.groupby(["dataset", "alpha"])}
check_true("symmetric = Marcialis thresholds in the three significant settings", all(same[(DS[d], a)] for d, a in sym),
           "in all of which symmetric rejection chose the Marcialis thresholds")
red = [-stat(d, a, "S1-Marcialis").mean_diff / val(d, a, "S1-Marcialis") for d, a in mar]
check("Marcialis reduction min %", min(red), "13", "by 13\\%--35\\%", scale=100); check("Marcialis reduction max %", max(red), "35", scale=100)
check_true("abstract: 13%-35% in five of six settings", True, None if SPRINGER else "by 13\\%--35\\% in five of six settings on the two larger subsets")   # IJIS v02: abstract pools original and new splits (checked below)
dr = pr[pr.method == "S4-Direct"]
check("direct search min Holm p", dr.p_holm.min(), "0.06", "Holm-adjusted $p\\ge0.06$")
check_true("direct search never significant", (dr.p_holm >= 0.05).all(), IJ("did not differ significantly from the direct empirical search in any setting", "No significant difference from the direct empirical search was detected in any setting"))
big = dr.loc[dr.mean_diff.idxmax()]
check("largest difference to direct search", big.mean_diff, "0.020", IJ("the largest difference, 0.020 in favor of the direct search, occurred on D3", "The largest difference, 0.020 in favor of the direct search, occurred on D3"))
check_true("... on D3 at 1e-4", big.dataset == DS["D3"] and np.isclose(big.alpha, 1e-4))
check_true("largest |difference| is that one", dr.mean_diff.abs().max() == big.mean_diff)
sp = [(val(d, a, "LR-P1-N2") - val(d, a, "S3-SPRT")) / val(d, a, "LR-P1-N2") for d in ["D2", "D3"] for a in [1e-3, 1e-4]]
check_true("SPRT lower and significant in all four D2/D3 alpha<=1e-3 settings",
           all(stat(d, a, "S3-SPRT").p_holm < 0.05 and stat(d, a, "S3-SPRT").mean_diff > 0 for d in ["D2", "D3"] for a in [1e-3, 1e-4]))
check("SPRT advantage min %", min(sp), "7", "reached 7\\%--13\\% lower FRR than the proposed design on D2 and D3", scale=100)
check("SPRT advantage max %", max(sp), "13", scale=100)
par = [(d, a, m) for d in ["D2", "D3"] for a in [1e-3, 1e-4] for m in ["P0-Parallel", "P1-LLR", "P2-LogReg"]]
psig = [(d, a, m) for d, a, m in par if stat(d, a, m).p_holm < 0.05 and stat(d, a, m).mean_diff > 0]
check("parallel significant comparisons", len(psig), "10", "by 7\\%--13\\%, in 10 of the 12 corresponding comparisons")
pv = [(val(d, a, "LR-P1-N2") - val(d, a, m)) / val(d, a, "LR-P1-N2") for d, a, m in psig]
check("parallel advantage min %", min(pv), "7", scale=100); check("parallel advantage max %", max(pv), "13", scale=100)
check("abstract: SPRT/parallel min %", min(sp + pv), "7", IJ("reached 7\\%--13\\% lower FRR at $\\far\\le10^{-3}$", "were significantly better, by 7\\%--13\\%, in 14 of 16 comparisons"), scale=100)
check("abstract: SPRT/parallel max %", max(sp + pv), "13", scale=100)
check("conclusion: 7%-13%", max(sp + pv), "13", IJ("achieved 7\\%--13\\% lower FRR", "achieved significantly lower FRR on these subsets, by 7\\%--13\\%, in 14 of 16 comparisons"), scale=100)
check("SPRT + parallel significant (D2/D3, alpha <= 1e-3) out of 16", len(sp) + len(psig), "14", None if not SPRINGER else "in 14 of 16 comparisons")
check_true("16 = 4 SPRT + 12 parallel comparisons", len(sp) + len(par) == 16)
sg = sy[sy.method == "LR-P1-N2"].stages_gen
if SPRINGER:   # IJIS v02: range restricted to the settings of the sentence (D2 and D3, alpha <= 1e-3)
    sg = sy[(sy.method == "LR-P1-N2") & sy.dataset.isin([DS["D2"], DS["D3"]]) & (sy.alpha <= 1e-3)].stages_gen
check("stages per genuine claim min", sg.min(), IJ("1.10", "1.17"), IJ("against 1.10--1.29 modalities per genuine claim", "against 1.17--1.29 modalities per genuine claim")); check("stages per genuine claim max", sg.max(), "1.29")
check_true("no significant difference on D1", (pr[pr.dataset == DS["D1"]].p_holm.dropna() >= 0.05).all(), "no difference was significant")
llr1 = pr[(pr.dataset == DS["D1"]) & (pr.method == "P1-LLR")]
check_true("LLR lower in 9 of 10 D1 splits at both alpha", (llr1.n_worse == 9).all() and len(llr1) == 2, "LLR fusion had the lower FRR in 9 of the 10 splits at both $\\alpha$")
fam = st[(st.family == "secondary") & st.method.isin(["MONO", "MONO-cont", "DE-N2", "NLS-N2", "LR-P1-N3", "LR-P2-N2"])]
check("envelope families max |diff|", fam.mean_diff.abs().max(), IJ("0.001", "0.0014"), IJ("by at most 0.001 in mean FRR", "by at most 0.0014 in mean FRR"))
ci = lambda ds: max(fam[fam.dataset == DS[ds]].ci_lo.abs().max(), fam[fam.dataset == DS[ds]].ci_hi.abs().max())
check_true("CI within 0.011 on D1", ci("D1") <= 0.011, "within $\\pm0.011$ on D1"); check_true("CI within 0.006 on D2, D3", max(ci("D2"), ci("D3")) <= 0.006, "$\\pm0.006$ on D2 and D3")
rl = st[(st.family == "secondary") & (st.method == "LR-P1-N2-rel")]
check("relative variant max |diff|", rl.mean_diff.abs().max(), "0.008", "differed by up to 0.008 in either direction")
check_true("relative variant both directions, extremes at 1e-4 on D2/D3", rl.mean_diff.max() > 0 > rl.mean_diff.min() and
           set(rl.loc[[rl.mean_diff.idxmax(), rl.mean_diff.idxmin()], "dataset"]) == {DS["D2"], DS["D3"]} and np.allclose(rl.loc[[rl.mean_diff.idxmax(), rl.mean_diff.idxmin()], "alpha"], 1e-4))
mm = st[(st.family == "secondary") & (st.method == "MAXMONO-K3") & (st.dataset == DS["D2"]) & np.isclose(st.alpha, 1e-4)].iloc[0]
check("MAXMONO D2 1e-4 diff", mm.mean_diff, "-0.019", "difference $-0.019$, CI $[-0.030,-0.009]$")
check("MAXMONO CI lo", mm.ci_lo, "-0.030"); check("MAXMONO CI hi", mm.ci_hi, "-0.009")

print("=" * 30, "Conservativeness (Section VI-D, discussion)")
rc = tab("T_rev_conservativeness"); oc = tab("T_conservativeness"); oc = oc[oc.method == "LR-P1-N2"]
al = rc[rc.chain == "all"]; seln = rc[rc.chain == "selected"]
check_true("all calibrated designs meet alpha on training", (al.far_ok_train == 1).all(), "every calibrated design met $\\alpha$ on its training half")
b1 = al[al.dataset == DS["D1"]].far_ok_test
check("bootstrap D1 compliance min %", b1.min(), "93", "93\\%--94\\% on D1 and 100\\% on D2", scale=100); check("bootstrap D1 compliance max %", b1.max(), "94", scale=100)
check("bootstrap D2 compliance %", al[al.dataset == DS["D2"]].far_ok_test.min(), "100", scale=100)
cp = oc[(oc.regime == "calibrated") & (oc.chain == "all")]; pl = oc[(oc.regime == "plain") & (oc.chain == "all")]
c1 = cp[cp.dataset == DS["D1"]].far_ok_test; c2 = cp[cp.dataset == DS["D2"]].far_ok_test; c3 = cp[cp.dataset == DS["D3"]].far_ok_test
check("CP D1 min %", c1.min(), "82", "against 82\\%--83\\% and 92.5\\%--100\\%", scale=100); check("CP D1 max %", c1.max(), "83", scale=100)
check("CP D2 min %", c2.min(), "92.5", scale=100); check("CP D2 max %", c2.max(), "100", scale=100)
u1 = pl[pl.dataset == DS["D1"]].far_ok_test
check("uncalibrated D1 min %", u1.min(), "37", "37\\%--60\\% (D1) without calibration", scale=100); check("uncalibrated D1 max %", u1.max(), "60", scale=100)
b3 = al[al.dataset == DS["D3"]]
check("bootstrap D3 min %", b3.far_ok_test.min(), "65", "reached only 65\\%--70\\% (35\\%--55\\%", scale=100); check("bootstrap D3 max %", b3.far_ok_test.max(), "70", scale=100)
check("CP D3 min %", c3.min(), "35", scale=100); check("CP D3 max %", c3.max(), "55", scale=100)
check("D3 median FAR/alpha min", b3.far_ratio_median.min(), "0.90", "median test FAR of 0.90--0.95"); check("D3 median FAR/alpha max", b3.far_ratio_median.max(), "0.95")
from scipy.stats import norm
check("benchmark compliance of a correct 95% bound %", norm.cdf(norm.ppf(0.95) / np.sqrt(2)), "88", "A correct 95\\% bound yields only about 88\\% compliance", scale=100)
cdg = tab("T_rev_calib_diag"); c3 = cdg[cdg.dataset == DS["D3"]]; c3p = c3[c3.method == "LR-P1-N2"]
check("D3 bootstrap SD min", c3p.boot_sd_mean.min(), "0.05", "bootstrap standard deviation of the training FAR (0.05$\\alpha$--0.13$\\alpha$)"); check("D3 bootstrap SD max", c3p.boot_sd_mean.max(), "0.13")
check_true("bootstrap SD matches split-to-split spread (within 0.02 alpha)", (abs(c3p.boot_sd_mean - c3p.shift_sd_over_sqrt2) <= 0.02).all(), IJ("matched the split-to-split spread", "matched the split-to-split spread of the difference between test and training FAR, so there is no evidence that the variance was underestimated"))
check("D3 shift proposed min", c3p.shift_mean.min(), "0.04", IJ("exceeded the training FAR on average by 0.04$\\alpha$--0.12$\\alpha$ for the proposed design and by up to 0.21$\\alpha$ for the other methods", "(0.04$\\alpha$--0.12$\\alpha$ for the proposed design and up to 0.21$\\alpha$ for the others)"))
check("D3 shift proposed max", c3p.shift_mean.max(), "0.12"); check("D3 shift others max", c3[c3.method != "LR-P1-N2"].shift_mean.max(), "0.21")
check_true("D3 shift positive for every method", (c3.shift_mean > 0).all())
check_true("no shift on D2", (cdg[cdg.dataset == DS["D2"]].shift_mean <= 0).all(), None if SPRINGER else "a shift absent on D2")
selp = tab("T_rev_selected"); selp = selp[selp.method == "LR-P1-N2"]
rat = (selp.frr_pred / selp.frr_test.replace(0, np.nan)).groupby([selp.dataset, selp.alpha]).median()
check("D1 prediction ratio min", rat[DS["D1"]].min(), "5.0", "on D1 it was 5.0--6.6 times the realized FRR (medians)"); check("D1 prediction ratio max", rat[DS["D1"]].max(), "6.6")
check_true("D3 prediction too low at alpha<=1e-3", (rat[DS["D3"]].loc[[1e-4, 1e-3]] < 1).all(), "too low on D3")
check("discussion: D1 ratio range", rat[DS["D1"]].max(), "6.6", "it was 5.0--6.6 times the realized FRR on D1 and too low on D3")
check("selected D1 prediction coverage %", seln[seln.dataset == DS["D1"]].frr_ok_test.min(), "100", "bounded the test FRR in all D1 splits", scale=100)
s2 = seln[seln.dataset == DS["D2"]].frr_ok_test
check("selected D2 coverage min %", s2.min(), "75", "in 75\\%--100\\% of the D2 splits", scale=100); check("selected D2 coverage max %", s2.max(), "100", scale=100)
s3 = seln[(seln.dataset == DS["D3"]) & (seln.alpha <= 1e-3)]
check_true("selected D3 coverage 0 at alpha<=1e-3", (s3.frr_ok_test == 0).all(), "and it failed in all D3 splits at $\\alpha\\le10^{-3}$")
check_true("selected D3 chains all correlated pairs", (s3.selected_chain_types == "correlated-pair").all(), "every selected chain combined the two correlated face matchers")
p1c = tab("T_prop1_check"); cm = p1c[p1c.dominance == "corner (main)"]; sp_ = p1c[(p1c.dominance == "sample points (ablation)") & (p1c.method == "LR-P1-N2")].iloc[0]
check("corner-dominance designs", cm.n_designs.sum(), "2044", IJ("for all 2044 single-stage designs", "for all 2044 feasible single-stage designs with corner dominance (all envelope families")); check_true("all satisfy prediction", (cm.frac_realized_le_pred == 1).all())
check("sampled-point dominance share %", sp_.frac_realized_le_pred, "88.75", "only 88.75\\% did", scale=100)
check("sampled-point max excess", sp_.max_excess, "0.0055", "largest excess 0.0055")
dc = tab("T_rev_decomposition"); d3 = dc[np.isclose(dc.alpha, 1e-3)].set_index("dataset")
check_true("envelope term dominates on D1", abs(d3.loc[DS["D1"], "envelope_median"]) > max(abs(d3.loc[DS["D1"], c]) for c in ["series bound_median", "dependence_median", "generalization_median"]),
           "envelope overestimation dominated on D1")
check_true("dependence largest on D3", d3.dependence_median.idxmax() == DS["D3"], "dependence between matchers raised the realized FRR most on D3")

print("=" * 30, "Security and cost (Section VI-E)")
if not SPRINGER:
    spf = tab("T_rev_spoof"); sv = lambda d, m, c="spoof_max": spf[(spf.dataset == DS[d]) & (spf.method == m)][c].iloc[0]
    d1 = spf[spf.dataset == DS["D1"]].set_index("method").spoof_max
    check("spoof proposed D1", d1["LR-P1-N2"], "0.85", "accepted 0.85 of the attempts on D1, the highest rate of the eight methods (0.57--0.82 for the others)")
    check_true("proposed highest on D1", d1.idxmax() == "LR-P1-N2")
    check("spoof others D1 min", d1.drop("LR-P1-N2").min(), "0.57"); check("spoof others D1 max", d1.drop("LR-P1-N2").max(), "0.82")
    d2s = spf[spf.dataset == DS["D2"]].spoof_max
    check("spoof proposed D2", sv("D2", "LR-P1-N2"), "0.86", "and 0.86 on D2, where all methods lay between 0.83 and 0.86")
    check("spoof D2 min", d2s.min(), "0.83"); check("spoof D2 max", d2s.max(), "0.86")
else:   # IJIS: per-trait perfect spoof (spoof_trait.py); a face artifact replaces both face scores
    spt = tab("T_rev_spoof_trait"); pv = lambda d: spt[spt.dataset == DS[d]].set_index("method").spoof_trait_max
    d1 = pv("D1"); check("trait spoof proposed D1", d1["LR-P1-N2"], "0.86", "accepted 0.86 of the attempts on D1, the highest rate of the eight methods (0.74--0.82 for the others)")
    check_true("trait spoof: proposed highest on D1", d1.idxmax() == "LR-P1-N2")
    check("trait spoof others D1 min", d1.drop("LR-P1-N2").min(), "0.74"); check("trait spoof others D1 max", d1.drop("LR-P1-N2").max(), "0.82")
    d2 = pv("D2"); check("trait spoof proposed D2", d2["LR-P1-N2"], "0.86", "0.86 on D2, where all methods lay between 0.83 and 0.86")
    check("trait spoof D2 min", d2.min(), "0.83"); check("trait spoof D2 max", d2.max(), "0.86")
    d3 = pv("D3"); check("trait spoof proposed D3", d3["LR-P1-N2"], "0.83", "and 0.83 on D3, where all methods lay between 0.81 and 0.86")
    check("trait spoof D3 min", d3.min(), "0.81"); check("trait spoof D3 max", d3.max(), "0.86")
    spm = tab("T_rev_spoof"); par3 = spm[(spm.dataset == DS["D3"]) & spm.method.str.startswith("P")].spoof_max
    check("per-algorithm spoof, parallel fusion D3 min", par3.min(), "0.30", "Parallel fusion accepted only 0.30--0.60 of the attempts on D3 when a single face algorithm was deceived")
    check("per-algorithm spoof, parallel fusion D3 max", par3.max(), "0.60")
    check_true("trait spoof: on D1 and D2 every method accepts most attempts of its worst trait", (spt[spt.dataset.isin([DS["D1"], DS["D2"]])].spoof_trait_max > 0.5).all(), "Every design combining two traits or fingers accepted most perfect spoofs of one of them")
    check_true("trait spoof: on D1, D2 and D4 every method accepts most attempts of its worst trait (conclusion)",
               (spt[spt.dataset.isin([DS["D1"], DS["D2"], "lfw_x_fing"])].spoof_trait_max > 0.5).all() and (spt.dataset == "lfw_x_fing").sum() == 8,
               "On the subsets that combine two traits or two fingers, every serial and parallel design accepted most perfect artifacts of its most vulnerable trait")
    rj_ = tab("T_reject_far"); r3_ = rj_[np.isclose(rj_.alpha, 1e-3) & rj_.dataset.isin([DS["D2"], DS["D3"]])].a_rej_median
    check_true("reject median stage FAR above 0.99 on D2 and D3 at 1e-3", (r3_ > 0.99).all(), "above 0.99 on D2 and D3")
rj = tab("T_reject_far"); acc = rj[np.isclose(rj.alpha, 1e-3)].a_acc_median
check("median accept FAR min (x1e-4)", acc.min() * 1e4, "3.6", "at a median stage FAR of $3.6\\times10^{-4}$--$5.0\\times10^{-4}$, where at least 64\\% of genuine scores")
check("median accept FAR max (x1e-4)", acc.max() * 1e4, "5.0")
check("min genuine acceptance at FAR 1e-4 %", (1 - dst["FRR@FAR<=0.0001"]).min(), "64", scale=100)
si = sy[sy.method == "LR-P1-N2"].groupby("dataset").stages_imp
check("impostor stages D1 min", si.min()[DS["D1"]], "2.8", "on average 2.8 (D1), 2.0 (D2), and 1.1--2.0 (D3) stages"); check("impostor stages D1 max", si.max()[DS["D1"]], "2.8")
check("impostor stages D2 min", si.min()[DS["D2"]], "2.0"); check("impostor stages D2 max", si.max()[DS["D2"]], "2.0")
check("impostor stages D3 min", si.min()[DS["D3"]], "1.1"); check("impostor stages D3 max", si.max()[DS["D3"]], "2.0")
r13 = rj[(rj.dataset == DS["D1"]) & np.isclose(rj.alpha, 1e-3)].iloc[0]
check("reject FAR median D1 1e-3", r13.a_rej_median, "0.84", IJ("(median 0.84 on D1 at $\\alpha=10^{-3}$)", "the reject thresholds lay at a median stage FAR of 0.84 on D1 and above 0.99 on D2 and D3"))
check("reject FAR > 0.1 share %", r13.frac_a_rej_gt_0p1, "99.5", "because 99.5\\% of the reject thresholds lay above", scale=100)
e8 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E8/cost_*.csv")], ignore_index=True)
gp8 = e8[e8.method == "GP stage cap"]; sp8 = e8[e8.method == "SPRT stage cap"]
capd = gp8[np.isfinite(gp8.param)].groupby(["dataset", "param"]).stages_imp.mean().reset_index()
check_true("mean realized impostor stages exceed kappa by at most 1e-4", (capd.stages_imp - capd.param).max() <= 1e-4 and gp8.feasible.all(),
           "mean realized impostor stages never exceeded $\\kappa$ by more than $10^{-4}$")
fr = lambda d, m, k: d[(d.dataset == DS[m]) & (np.isclose(d.param, k) if np.isfinite(k) else ~np.isfinite(d.param))].frr_test.mean()
check("D2 FRR no cap", fr(gp8, "D2", np.inf), "0.0687", "$\\kappa=1.2$ raised the test FRR on D2 from 0.0687 to 0.0758")
check("D2 FRR kappa 1.2", fr(gp8, "D2", 1.2), "0.0758")
check("D2 SPRT kappa 1.1", fr(sp8, "D2", 1.1), "0.0843", "($\\kappa=1.1$: 0.0843 against 0.0789)"); check("D2 GP kappa 1.1", fr(gp8, "D2", 1.1), "0.0789")
cmp = [(ds, k, fr(gp8, d_, k), fr(sp8, d_, k)) for d_, ds in DS.items() for k in sorted(set(gp8[gp8.dataset == ds].param))]
wins = [(ds, k) for ds, k, g, q in cmp if g < q - 1e-12]
check_true("GP lower than SPRT only at the tightest D1 and D2 budgets", sorted(wins) == sorted([(DS["D1"], 1.3), (DS["D2"], 1.1)]), IJ("except at the tightest budgets on D1 ($\\kappa=1.3$: 0.0382 against 0.0282)", "reached lower FRR at every budget except the tightest ones on D1 ($\\kappa=1.3$: 0.0382 against 0.0282)"))
check("D1 SPRT kappa 1.3", fr(sp8, "D1", 1.3), "0.0382"); check("D1 GP kappa 1.3", fr(gp8, "D1", 1.3), "0.0282")

if SPRINGER:
    ct = tab("T_rev_cost_spoof_trait"); ct = ct[ct.method == "GP stage cap"]
    sw = ct.groupby(["dataset", "param"]).spoof_trait_worst.mean().round(2)
    rng_ = {d: sorted(set(sw[DS[d]].values)) for d in DS}
    check_true(f"IJIS: stage cap leaves GP trait-spoof acceptance at 0.87-0.88 (D1), 0.86 (D2), 0.83-0.84 (D3): {rng_}",
               rng_ == {"D1": [0.87, 0.88], "D2": [0.86], "D3": [0.83, 0.84]},
               "their worst-case spoof acceptance stayed within 0.87--0.88 on D1, at 0.86 on D2, and within 0.83--0.84 on D3 for every $\\kappa$")
    check_true("IJIS: Table 6 covers all feasible stage-capped designs", len(ct) == int(gp8.feasible.sum()))
    print("=" * 30, "IJIS revision claims")
    import json as _json
    e1h = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E1/main_*.csv")], ignore_index=True); e1h = e1h[e1h.method == "HYP"]
    Cs = e1h.terms.map(lambda t: _json.loads(t)[0][0])
    check_true("IJIS: HYP envelopes are single terms C a^-1", e1h.terms.map(lambda t: len(_json.loads(t)) == 1 and _json.loads(t)[0][1] == -1).all())
    check_true(f"IJIS: smallest hyperbola constant C_s = {Cs.min():.6f} > 0.0024", Cs.min() > 0.0024, "and $C_s>0.0024$ for every BSSR1 matcher and training half")
    check_true("IJIS: C_s > 1e-3 for all matchers and splits (HYP infeasible at alpha <= 1e-3)", (Cs > 1e-3).all())
    hy = sy[(sy.method == "HYP") & (sy.dataset == DS["D2"]) & np.isclose(sy.alpha, 1e-2)].iloc[0]
    check_true("IJIS: HYP feasible in six D2 splits at 1e-2", int(hy.n_seeds) == 6 or int(round(hy.feas_rate * 10)) == 6, "(on D2 over the six splits in which it was feasible)")
    v2x = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2v2/e2v2_*_N2.csv")], ignore_index=True)
    v2x = v2x[(v2x.method == "LR-BB exact dual") & (v2x.obj == "P1")]; shx = (v2x.ref_value - v2x.root_lb) / v2x.ref_value
    check_true("IJIS: P1 root bound within 1e-4 in 39, within 1e-6 in 38", int((shx <= 1e-4).sum()) == 39 and int((shx <= 1e-6).sum()) == 38,
               "(within a relative tolerance of $10^{-4}$; 38 within $10^{-6}$)")
    e7x = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E7v2/e7v2_*.csv")], ignore_index=True)
    n1 = e7x[(e7x.which == "N") & (e7x.N == 1) & (e7x.method == "LR-BB exact dual") & (e7x.obj == "P1")].set_index("dataset")
    check("IJIS: N=1 root gap D1 face C %", n1.loc[DS["D1"]].root_gap, "6.0", "the P1 root bound fell 6.0\\% short on D1 face C and 0.17\\% on D2 left index", scale=100)
    check("IJIS: N=1 root gap D2 left index %", n1.loc[DS["D2"]].root_gap, "0.17", scale=100)
    check_true("IJIS: N=1 needed 18 nodes", (n1.nodes == 18).all(), "and LR-BB needed 18 nodes")
    p2n = e7x[(e7x.which == "N") & (e7x.N >= 2) & (e7x.method == "LR-BB exact dual") & (e7x.obj == "P2")]
    dec = p2n.groupby("dataset").value.agg(lambda v: (v.max() - v.min()) / v.max())
    check("IJIS: P2 best value decrease N=2..5 max %", dec.max(), "0.4", "the best P2 values decreased by at most 0.4\\% (D1 face C)", scale=100)
    check_true("IJIS: ... on D1 face C", dec.idxmax() == DS["D1"])
    drd = pr[(pr.method == "S4-Direct") & (pr.dataset == DS["D3"]) & np.isclose(pr.alpha, 1e-4)].iloc[0]
    check("IJIS: direct D3 1e-4 CI lo", drd.ci_lo, "0.004", "its unadjusted 95\\% CI $[0.004,0.037]$ excludes zero"); check("IJIS: direct D3 1e-4 CI hi", drd.ci_hi, "0.037")
    check_true("IJIS: direct search lower in all ten splits", int(drd.n_worse) == 10, "occurred on D3 at $\\alpha=10^{-4}$ in all ten splits")
    fo = sy[(sy.dataset == DS["D3"]) & np.isclose(sy.alpha, 1e-4)].set_index("method").far_over_alpha
    check("IJIS: D3 1e-4 FAR/alpha proposed", fo["LR-P1-N2"], "0.90", "the mean test FAR was 0.90$\\alpha$ for the proposed design and 0.99$\\alpha$ for the direct search")
    check("IJIS: D3 1e-4 FAR/alpha direct", fo["S4-Direct"], "0.99")
    prx = pr[pr.p_corr_t.notna()].copy(); pv_ = prx.p_corr_t.values; o_ = np.argsort(pv_); m_ = len(pv_); adj_ = np.empty(m_); run_ = 0
    for r_, i_ in enumerate(o_): run_ = max(run_, (m_ - r_) * pv_[i_]); adj_[i_] = min(1, run_)
    prx["pg"] = adj_
    check_true("IJIS: 59 primary comparisons", m_ == 59, "applied across all 59 primary comparisons")
    sub = prx[prx.dataset != DS["D1"]]
    check_true("IJIS: global Holm - Marcialis 4 of 6 (D2, D3)", int((sub[sub.method == "S1-Marcialis"].pg < 0.05).sum()) == 4,
               "remained significantly better than the Marcialis rule in four of these six settings")
    s3 = sub[np.isclose(sub.alpha, 1e-3) | np.isclose(sub.alpha, 1e-4)]
    check_true("IJIS: global Holm - SPRT 3 of 4, parallel 6 of 12 (D2, D3, alpha <= 1e-3)",
               int((s3[s3.method == "S3-SPRT"].pg < 0.05).sum()) == 3 and len(s3[s3.method == "S3-SPRT"]) == 4
               and int((s3[s3.method.str.startswith("P")].pg < 0.05).sum()) == 6 and len(s3[s3.method.str.startswith("P")]) == 12,
               "remained significantly better in three of four and six of 12 comparisons")
    prv = pr[pr.p_corr_t.notna()]; fam_n = prv.groupby(["dataset", "alpha"]).size()
    hyp_ok = prv[prv.method == "HYP"].groupby(["dataset", "alpha"]).size()
    check_true("IJIS: HYP tested in 3 settings only", len(hyp_ok) == 3)
    check_true("IJIS: family has 8 tests where HYP feasible, 7 otherwise", all(fam_n[k] == (8 if k in hyp_ok.index else 7) for k in fam_n.index),
               "(seven where HYP is infeasible)")
    dep = sy[sy.method == "LR-P1-N2"].set_index(["dataset", "alpha"]).far_ok * 10
    check("IJIS: deployed compliance D1 1e-3", dep[(DS["D1"], 1e-3)], "8.67", "met $\\alpha$ in 8.67 and 9.9 of the 10 test halves on D1 (tie-averaged)")
    check("IJIS: deployed compliance D1 1e-2", dep[(DS["D1"], 1e-2)], "9.9")
    check_true("IJIS: deployed compliance D2 all, D3 5-7", all(dep[(DS["D2"], a)] == 10 for a in (1e-2, 1e-3, 1e-4))
               and sorted(round(dep[(DS["D3"], a)]) for a in (1e-2, 1e-3, 1e-4)) == [5, 6, 7], "and in 5 to 7 on D3")
    co = tab("T_rev_conservativeness"); co3 = co[co.dataset == DS["D3"]]
    check_true("IJIS: D3 single-matcher 70%, pairs 60-70%", set((co3[co3.chain == "single"].far_ok_test * 100).round()) == {70}
               and set((co3[co3.chain == "correlated-pair"].far_ok_test * 100).round()) == {60, 70}, "(70\\% against 60\\%--70\\%)")
    tt = cdg.shift_mean / (cdg.shift_sd * np.sqrt(1 + 1 / 10))
    check("IJIS: largest corrected t of the shift", tt.abs().max(), "1.13", "but it never exceeded 1.13 corrected standard errors")
    check("IJIS: D1 1e-3 shift proposed", cdg[(cdg.dataset == DS["D1"]) & np.isclose(cdg.alpha, 1e-3) & (cdg.method == "LR-P1-N2")].shift_mean.iloc[0], "0.18",
          "and for the proposed design on D1 at $\\alpha=10^{-3}$ (0.18$\\alpha$)")
    mono = fam[fam.method.isin(["MONO", "MONO-cont"])].mean_diff.abs().max()
    check("IJIS: single-monomial designs within 0.0014", mono, "0.0014", "the designs built on a single monomial were within 0.0014 of the proposed ones")
    es = open("envelope_solvers.py").read()
    check_true("IJIS: subgradient settings (mu=0 start, step factor 2, patience 15, 2 x 300 iterations)",
               "mu0 = np.zeros(" in es and "lam=2.0, patience=15" in es and "K_root=300" in es and es.count("node.run(root_iv") == 2,
               "starting from $\\mu=0$ with a step factor of 2 that was halved after 15 non-improving iterations")
    refx = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2ref/ref_*.csv")], ignore_index=True)
    bmin = refx[["b1", "b2"]].where(refx[["c1", "c2"]].values > 1e-12).min(axis=1); refx["bmin"] = bmin
    bm = refx.groupby("obj").bmin.min()
    check("IJIS: smallest optimal exponent P1", bm["P1"], "-1.62", "(the smallest was $-1.62$ for P1 and $-1.48$ for P2)"); check("IJIS: smallest optimal exponent P2", bm["P2"], "-1.48")
    p95 = cdg[cdg.method == "LR-P1-N2"].set_index("dataset").boot_p95_mean
    check("IJIS: D2 bootstrap p95 min", p95[DS["D2"]].min(), "0.81", "at 0.81$\\alpha$--0.91$\\alpha$, against about 1.00$\\alpha$ on D3"); check("IJIS: D2 bootstrap p95 max", p95[DS["D2"]].max(), "0.91")
    check_true("IJIS: D3 bootstrap p95 about 1.00", ((p95[DS["D3"]] > 0.99) & (p95[DS["D3"]] <= 1.0)).all())
    # ------------------------------------------------------------------ IJIS v02 (retitled revision): new analyses
    print("=" * 30, "IJIS v02 claims (title, MIQP, matched FAR, new splits, cross-fitting, D4)")
    mt_ = open(f"{P}/main.tex").read()
    check_true("v03: title", "Grid-Certified Posynomial FAR--FRR Envelopes for Serial Multibiometric Threshold Design via Lagrangian Relaxation-Based Branch-and-Bound" in mt_)
    refq = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2ref/ref_*.csv")], ignore_index=True); refq = refq[refq.obj == "P1"]
    mq2 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2miqp/miqp_*_N2.csv")], ignore_index=True)
    mq2 = mq2.merge(refq[["dataset", "seed", "matcher", "value"]].rename(columns={"value": "opt"}), on=["dataset", "seed", "matcher"])
    rd2 = ((mq2.value - mq2.opt) / mq2.opt).abs().max()
    check_true(f"v02: MIQP N=2 reproduces all 40 optima, max rel. diff {rd2:.2e} rounds to 3e-14",
               len(mq2) == 40 and (mq2.status == "optimal").all() and float(f"{rd2:.0e}") == 3e-14,
               "reproduced the reference optimum in all 40 instances (relative difference at most $3\\times10^{-14}$)")
    check_true(f"v02: MIQP gap <= 1e-4 (max {mq2.gap.max():.2e}) and at most 3 iterations", mq2.gap.max() <= 1e-4 and mq2.iters.max() == 3,
               "certified it within $10^{-4}$ after at most three iterations")
    tq = mq2.groupby("dataset").time.mean()
    check("v02: MIQP time min", tq.min(), "0.60", "(0.60--1.93~s per instance, subset means)"); check("v02: MIQP time max", tq.max(), "1.93")
    mq3 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2miqp/miqp_*_N3.csv")], ignore_index=True)
    x3 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2v2/e2v2_*_N3.csv")], ignore_index=True)
    x3 = x3[(x3.method == "LR-BB exact dual") & (x3.obj == "P1")]
    m3 = x3.merge(mq3[["dataset", "seed", "matcher", "value", "status"]].rename(columns={"value": "miqp"}), on=["dataset", "seed", "matcher"])
    rd3 = ((m3.value - m3.miqp) / m3.miqp).abs().max()
    check_true(f"v02: N=3 LR-BB matches MIQP in all 40 (max rel. diff {rd3:.2e})", len(m3) == 40 and (m3.status == "optimal").all(), "matched it in all 40 P1 instances")
    check("v02: N=3 max rel. diff (1e-8)", rd3 * 1e8, "2.4", "(relative difference at most $2.4\\times10^{-8}$)")
    rg3 = (m3.miqp - m3.root_lb) / m3.miqp
    check_true(f"v02: N=3 root bound closes the gap in every instance (max {rg3.max():.1e})", (rg3 <= 1e-4).all(), "its root bound closed the gap in every instance")
    m23 = m3.merge(mq2[["dataset", "seed", "matcher", "value"]].rename(columns={"value": "v2"}), on=["dataset", "seed", "matcher"])
    imp = (m23.v2 - m23.miqp) / m23.v2
    check_true("v02: three terms improve on two in one instance only", int((imp > 1e-6).sum()) == 1, "three terms improved on two in only one instance, by 0.26\\%")
    check("v02: N=3 improvement %", imp.max(), "0.26", scale=100)
    # matched FAR (Online Resource 1, Table S15)
    mfs = tab("T_rev_matchedfar_stats"); mfs = mfs[mfs.p_corr_t.notna()]
    pm_ = pr[pr.p_corr_t.notna()].merge(mfs, on=["dataset", "alpha", "method"], suffixes=("", "_m"))
    check_true("v02: matched FAR - 59 comparisons, all keep their sign", len(pm_) == 59 and (np.sign(pm_.mean_diff) == np.sign(pm_.mean_diff_m)).all(),
               "all 59 primary differences kept their sign")
    keep = (pm_.p_holm < 0.05) == (pm_.p_holm_m < 0.05); lost = pm_[~keep]
    check("v02: matched FAR - significance kept", int(keep.sum()), "54", "and 54 kept their significance decision")
    check_true("v02: matched FAR - the five lost significance, p to 0.05-0.09", len(lost) == 5 and (lost.p_holm < 0.05).all() and lost.p_holm_m.min() >= 0.05 and round(lost.p_holm_m.max(), 2) == 0.09,
               "the Holm-adjusted $p$ rose from below 0.05 to 0.05--0.09")
    mfd = mfs[(mfs.dataset == DS["D3"]) & np.isclose(mfs.alpha, 1e-4) & (mfs.method == "S4-Direct")].iloc[0]
    check("v02: matched FAR D3 1e-4 direct", mfd.mean_diff, "0.018", "the direct search remained lower by 0.018 in that setting")
    # D4 data (Table 1, Sect. 5.1-5.2)
    D4 = "lfw_x_fing"; d4 = dst[dst.dataset == D4].set_index("matcher")
    for m, eer, f3 in [("face_S", "0.007", "0.013"), ("li_V", "0.082", "0.180"), ("ri_V", "0.055", "0.124")]:
        check(f"v02: D4 {m} EER", d4.loc[m].EER, eer); check(f"v02: D4 {m} FRR@1e-3", d4.loc[m]["FRR@FAR<=0.001"], f3)
    check_true("v02: D4 1680 subjects", (d4.n_genuine == 1680).all(), "1680 /\\\\$2.8\\times10^{6}$")
    check("v02: D4 impostors 2.8e6", d4.n_impostor.iloc[0] / 1e6, "2.8")
    c4 = cor[cor.dataset == D4].set_index("pair").genuine_pearson
    ffc = sorted(round(v, 2) for k_, v in c4.items() if "face" in k_)
    check_true(f"v02: D4 face-finger genuine corr {ffc}", ffc == [-0.01, 0.02], "face--finger\\\\0.02, $-0.01$")
    check("v02: D4 L-R genuine corr", c4["li_V-ri_V"], "0.41")
    check_true("v02: D4 face FRR at 1e-3 quoted in Sect. 6.3", round(d4.loc["face_S"]["FRR@FAR<=0.001"], 3) == 0.013, "Its FRR at $\\far=10^{-3}$ is 0.013")
    check_true("v02: D4 test halves 840 genuine, 7.0e5 impostors", 1680 // 2 == 840 and round(840 * 839 / 1e5, 1) == 7.0,
               "its test halves contain 840 genuine and $7.0\\times10^{5}$ impostor comparisons")
    lg = open("../data/lfw_x_fing_build.json").read()
    check_true("v02: YuNet found a face in every image", '"face_detect_fail_ref": 0' in lg and '"face_detect_fail_probe": 0' in lg, "the detector found a face in every image")
    # new splits (Online Resource 1, Table S16; Sect. 6.3)
    fsy = tab("T_fresh_system"); fst = tab("T_fresh_stats"); fco = tab("T_fresh_compliance")
    fv = lambda d, a, m, cal="boot", c="frr_test": fsy[(fsy.dataset == DS.get(d, D4)) & np.isclose(fsy.alpha, a) & (fsy.method == m) & (fsy.calib == cal)][c].iloc[0]
    fs = lambda d, a, m, cal="boot": fst[(fst.dataset == DS.get(d, D4)) & np.isclose(fst.alpha, a) & (fst.method == m) & (fst.calib == cal)].iloc[0]
    check_true("v02: 20 new splits of D1-D3, 10 of D4, 70 files", fsy[fsy.dataset != D4].n_seeds.max() == 20 and fsy[fsy.dataset == D4].n_seeds.max() == 10
               and len(glob.glob(f"{R}/E3fresh/fresh_*.csv")) == 70, "repeated on 20 further random halvings of the same D1--D3 subjects (seeds 10--29)")
    mar2 = [(d, a) for d in ["D2", "D3"] for a in [1e-2, 1e-3, 1e-4] if fs(d, a, "S1-Marcialis").p_holm < 0.05 and fs(d, a, "S1-Marcialis").mean_diff < 0]
    red2 = [-fs(d, a, "S1-Marcialis").mean_diff / fv(d, a, "S1-Marcialis") for d, a in mar2]
    check("v02: new splits - Marcialis significant settings", len(mar2), "5", "again had significantly lower FRR than the Marcialis rule in five of the six D2 and D3 settings, by 12\\%--36\\%")
    check("v02: Marcialis reduction min %", min(red2), "12", scale=100); check("v02: Marcialis reduction max %", max(red2), "36", scale=100)
    check("v02: sixth setting p", fs("D3", 1e-4, "S1-Marcialis").p_holm, "0.07", "(Holm-adjusted $p=0.07$ in the sixth, D3 at $\\alpha=10^{-4}$)")
    sp2 = [(d, a, m) for d in ["D2", "D3"] for a in [1e-3, 1e-4] for m in ["S3-SPRT", "P0-Parallel", "P1-LLR", "P2-LogReg"]]
    sg2 = [(d, a, m) for d, a, m in sp2 if fs(d, a, m).p_holm < 0.05 and fs(d, a, m).mean_diff > 0]
    adv2 = [(fv(d, a, "LR-P1-N2") - fv(d, a, m)) / fv(d, a, "LR-P1-N2") for d, a, m in sg2]
    pv_orig = [(val(d, a, "LR-P1-N2") - val(d, a, m)) / val(d, a, "LR-P1-N2") for d, a, m in psig]   # original splits (pv is reused above)
    check("v02: new splits - SPRT/parallel significant of 16", len(sg2), "14", "by 9\\%--13\\%, in 14 of the 16 comparisons at $\\alpha\\le10^{-3}$")
    check("v02: SPRT/parallel adv min %", min(adv2), "9", scale=100); check("v02: SPRT/parallel adv max %", max(adv2), "13", scale=100)
    check_true("v02: pooled abstract ranges 12-36 (Marcialis) and 7-13 (SPRT/parallel)",
               round(100 * min(red + red2)) == 12 and round(100 * max(red + red2)) == 36 and round(100 * min(sp + pv_orig + adv2)) == 7 and round(100 * max(sp + pv_orig + adv2)) == 13,
               "significantly reduced the FRR of the zero-FAR/zero-FRR rule by 12\\%--36\\% in five of six settings on the two larger subsets")
    check_true("v02: 14 of 16 on both sets of splits", len(sp) + len(psig) == 14 and len(sg2) == 14, "in 14 of 16 comparisons on both sets of splits")
    check_true("v02: new splits - nothing significant on D1", (fst[(fst.dataset == DS["D1"]) & (fst.calib == "boot")].p_holm.dropna() >= 0.05).all(), "and no difference on D1 was significant")
    dr2 = fst[(fst.method == "S4-Direct") & (fst.calib == "boot") & (fst.dataset != D4)]; dsig = dr2[dr2.p_holm < 0.05]
    check_true("v02: direct search significant only on D3 at 1e-4, lower in all 20", len(dsig) == 1 and dsig.dataset.iloc[0] == DS["D3"] and np.isclose(dsig.alpha.iloc[0], 1e-4) and int(dsig.n_worse.iloc[0]) == 20,
               "on D3 at $\\alpha=10^{-4}$ it had lower FRR in all 20 splits")
    check("v02: direct D3 1e-4 diff", dsig.mean_diff.iloc[0], "0.018", "by 0.018 (95\\% CI $[0.004,0.031]$)")
    check("v02: direct CI lo", dsig.ci_lo.iloc[0], "0.004"); check("v02: direct CI hi", dsig.ci_hi.iloc[0], "0.031")
    check("v02: other direct p min", dr2[dr2.p_holm >= 0.05].p_holm.min(), "0.31", "(Holm-adjusted $p\\ge0.31$)", tol=0.005)
    check_true("v02: direct search never significant on original splits or D4", (pr[pr.method == "S4-Direct"].p_holm >= 0.05).all()
               and (fst[(fst.dataset == D4) & (fst.method == "S4-Direct")].p_holm >= 0.05).all(), "differs significantly from a direct empirical search in only one setting")
    # sensitivity of the Marcialis comparison (Sect. 6.3, abstract, conclusion) and matched FAR on the new splits and D4 (Table S18)
    mfo = mfs[(mfs.method == "S1-Marcialis") & mfs.dataset.isin([DS["D2"], DS["D3"]])]
    check("v02: Marcialis significant at matched FAR, original splits", int(((mfo.p_holm < 0.05) & (mfo.mean_diff < 0)).sum()), "4",
          "at matched FAR it held in four of the six D2 and D3 settings of the original splits")
    check("v02: D3 1e-3 matched p", mfo[(mfo.dataset == DS["D3"]) & np.isclose(mfo.alpha, 1e-3)].p_holm.iloc[0], "0.06", "(on D3 at $\\alpha=10^{-3}$ the Holm-adjusted $p$ rose to 0.06)")
    mfn_ = tab("T_fresh_matchedfar_stats"); mfn_ = mfn_[mfn_.p_corr_t.notna()]
    mfm = mfn_[(mfn_.method == "S1-Marcialis") & mfn_.dataset.isin([DS["D2"], DS["D3"]])]
    check("v02: Marcialis significant at matched FAR, new splits", int(((mfm.p_holm < 0.05) & (mfm.mean_diff < 0)).sum()), "5", "and in five of six of the new splits")
    xm_ = fst[(fst.calib == "xfit") & (fst.method == "S1-Marcialis") & fst.dataset.isin([DS["D2"], DS["D3"]])]
    check("v02: Marcialis significant with held-out calibration", int(((xm_.p_holm < 0.05) & (xm_.mean_diff < 0)).sum()), "4", "and with the held-out calibration in four of six")
    xs_ = fst[(fst.calib == "xfit") & fst.dataset.isin([DS["D2"], DS["D3"]]) & (fst.alpha <= 1e-3) & fst.method.isin(["S3-SPRT", "P0-Parallel", "P1-LLR", "P2-LogReg"])]
    check("v02: held-out SPRT/parallel significant of 16", int(((xs_.p_holm < 0.05) & (xs_.mean_diff > 0)).sum()), "12",
          "while the SPRT and parallel fusion remained significantly better in 12 of the 16 comparisons at $\\alpha\\le10^{-3}$")
    check_true("v02: ... of 16", len(xs_) == 16)
    gh = pr[pr.p_corr_t.notna()].copy(); pv_g = gh.p_corr_t.values; og = np.argsort(pv_g); mg = len(pv_g); ag = np.empty(mg); rn = 0
    for r_, i_ in enumerate(og): rn = max(rn, (mg - r_) * pv_g[i_]); ag[i_] = min(1, rn)
    gh["pg"] = ag; ghm = gh[(gh.method == "S1-Marcialis") & (gh.dataset != DS["D1"])]
    check_true("v02: abstract 'four of six in most sensitivity analyses' (global Holm 4, matched original 4, matched new 5, held-out 4)",
               int((ghm.pg < 0.05).sum()) == 4 and int(((mfo.p_holm < 0.05) & (mfo.mean_diff < 0)).sum()) == 4
               and int(((mfm.p_holm < 0.05) & (mfm.mean_diff < 0)).sum()) == 5 and int(((xm_.p_holm < 0.05) & (xm_.mean_diff < 0)).sum()) == 4,
               "(four of six in most sensitivity analyses)")
    bb_ = fst[(fst.calib == "boot") & fst.p_corr_t.notna()].merge(mfn_, on=["dataset", "alpha", "method"], suffixes=("", "_m"))
    check("v02: matched FAR new splits - comparisons", len(bb_), "82", "80 of the 82 differences kept their sign and 77 their significance decision")
    check("v02: ... kept sign", int((np.sign(bb_.mean_diff) == np.sign(bb_.mean_diff_m)).sum()), "80")
    check("v02: ... kept significance", int(((bb_.p_holm < 0.05) == (bb_.p_holm_m < 0.05)).sum()), "77")
    dmn = mfn_[(mfn_.dataset == DS["D3"]) & np.isclose(mfn_.alpha, 1e-4) & (mfn_.method == "S4-Direct")].iloc[0]
    check_true("v02: matched FAR direct D3 1e-4 on 19 splits", int(dmn.n) == 19, "on the 19 splits in which its matched FAR could be reached")
    check("v02: matched FAR new splits - direct D3 1e-4 p", dmn.p_holm, "0.11", "the advantage of the direct search on D3 at $\\alpha=10^{-4}$ was no longer significant (Holm-adjusted $p=0.11$")
    msy = tab("T_fresh_matchedfar_system"); m4 = msy[msy.dataset == D4]
    pm4 = m4[(m4.method == "LR-P1-N2") & (m4.alpha >= 1e-3)].frr_matched; dm4 = m4[m4.method.isin(["S1-Marcialis", "S2-Symmetric", "S4-Direct"]) & (m4.alpha >= 1e-3)].frr_matched
    s4m = mfn_[(mfn_.dataset == D4) & (mfn_.p_holm < 0.05)]
    check_true("v02: D4 matched FAR - proposed not lower than decision-level rules at alpha>=1e-3; only sum fusion significant",
               pm4.min() > dm4.max() and set(s4m.method) == {"P0-Parallel"},
               "at matched FAR, the proposed design still had higher mean FRR than the three decision-level rules at $\\alpha=10^{-2}$ and $10^{-3}$, and again only sum fusion differed significantly from it")
    cnt = lambda m, a: fv("D4", a, m, c="far_ok")
    fewer = [(m, a) for m in ["S1-Marcialis", "S2-Symmetric", "S4-Direct", "S3-SPRT", "P0-Parallel", "P1-LLR", "P2-LogReg"] for a in (1e-2, 1e-3, 1e-4) if cnt(m, a) < cnt("LR-P1-N2", a) - 1e-9]
    check_true(f"v02: most D4 baselines met alpha in fewer halves ({len(fewer)} of 21)", len(fewer) > 21 / 2, "Most baselines also met $\\alpha$ in fewer test halves than the proposed design")
    # D4 system results (Sect. 6.3)
    sg4 = fsy[(fsy.dataset == D4) & (fsy.calib == "boot") & (fsy.method == "LR-P1-N2")]
    check("v02: D4 proposed stages min", sg4.stages_gen.min(), "1.02", "(1.02--1.05 modalities per genuine claim)"); check("v02: D4 proposed stages max", sg4.stages_gen.max(), "1.05")
    check("v02: D4 proposed FRR min", sg4.frr_test.min(), "0.0023", "reached a test FRR of 0.0023--0.0042"); check("v02: D4 proposed FRR max", sg4.frr_test.max(), "0.0042")
    lk = fsy[(fsy.dataset == D4) & (fsy.calib == "boot") & fsy.method.isin(["S3-SPRT", "P1-LLR", "P2-LogReg"])].frr_test
    check("v02: D4 SPRT/LLR/LogReg min", lk.min(), "0.0007", "This was higher than that of the SPRT and the LLR and logistic-regression fusions (0.0007--0.0011)"); check("v02: D4 SPRT/LLR/LogReg max", lk.max(), "0.0011")
    orr = fsy[(fsy.dataset == D4) & (fsy.calib == "boot") & fsy.method.isin(["S1-Marcialis", "S2-Symmetric", "S4-Direct"])].frr_test
    orr3 = fsy[(fsy.dataset == D4) & (fsy.calib == "boot") & (fsy.alpha >= 1e-3) & fsy.method.isin(["S1-Marcialis", "S2-Symmetric", "S4-Direct"])].frr_test
    check("v02: D4 decision-level rules alpha>=1e-3 min", orr3.min(), "0.0011", "than that of the three decision-level rules (0.0011--0.0020)"); check("v02: D4 decision-level max", orr3.max(), "0.0020")
    p4 = fsy[(fsy.dataset == D4) & (fsy.calib == "boot") & (fsy.alpha >= 1e-3) & (fsy.method == "LR-P1-N2")].frr_test
    check_true("v02: D4 proposed higher than every decision-level rule at alpha>=1e-3", p4.min() > orr3.max(), "On D4 it had higher mean FRR than the three decision-level rules at $\\alpha=10^{-2}$ and $10^{-3}$, without significant differences.")
    check("v02: D4 one error", 1 / 840, "0.0012", "one error changes the FRR by 0.0012")
    s4 = fst[(fst.dataset == D4) & (fst.calib == "boot") & (fst.p_holm < 0.05)]
    check_true("v02: D4 only sum fusion significant (1e-3, 1e-4), proposed lower", set(s4.method) == {"P0-Parallel"} and len(s4) == 2 and (s4.mean_diff < 0).all(),
               "whose fingerprint scores outweigh the face score and whose FRR was 0.0171 and 0.0388 at $\\alpha\\le10^{-3}$, were significant")
    check("v02: D4 sum 1e-3", fv("D4", 1e-3, "P0-Parallel"), "0.0171"); check("v02: D4 sum 1e-4", fv("D4", 1e-4, "P0-Parallel"), "0.0388")
    nb = fst[(fst.dataset == D4) & (fst.calib == "boot") & fst.method.isin(["S3-SPRT", "P1-LLR"])].n_better
    check_true("v02: D4 SPRT and LLR never worse than proposed", (nb == 0).all(), "the SPRT and LLR fusion had lower FRR than the proposed design in every split in which they differed")
    check_true("v02: D4 - only sum fusion significant (Discussion)", set(s4.method) == {"P0-Parallel"}, "on D4 only sum fusion differed significantly from the proposed design, although the SPRT and LLR fusion were never worse in any split")
    hc = tab("T_d4_hyp_constants"); hf = hc[hc.matcher == "face_S"].C
    check("v02: D4 face C_s min (1e-4)", hf.min() * 1e4, "4.8", "the constant $C_s$ of the hyperbolic face envelope was $4.8\\times10^{-4}$--$8.1\\times10^{-4}$ on the training halves"); check("v02: D4 face C_s max (1e-4)", hf.max() * 1e4, "8.1")
    check_true("v02: D4 HYP feasible in all splits at 1e-3, infeasible at 1e-4", fv("D4", 1e-3, "HYP", c="n_seeds") == 10 and
               len(fsy[(fsy.dataset == D4) & (fsy.calib == "boot") & (fsy.method == "HYP") & np.isclose(fsy.alpha, 1e-4)].dropna(subset=["frr_test"])) == 0)
    check("v02: D4 HYP 1e-3 FRR", fv("D4", 1e-3, "HYP"), "0.0074", "the hyperbolic model was feasible at $\\alpha=10^{-3}$ on D4, with a test FRR of 0.0074")
    ab4 = fsy[(fsy.dataset == D4) & (fsy.calib == "boot") & fsy.method.isin(["LR-P1-N2", "S1-Marcialis", "S2-Symmetric", "S4-Direct", "S3-SPRT"])].frr_test
    sl4 = fsy[(fsy.dataset == D4) & (fsy.calib == "boot") & fsy.method.isin(["S3-SPRT", "P1-LLR"])].frr_test
    check_true(f"v02: abstract - D4 SPRT/LLR about 0.001 ({sl4.min():.4f}-{sl4.max():.4f}), proposed 0.002-0.004, not significant",
               round(sl4.min(), 3) == 0.001 and round(sl4.max(), 3) == 0.001 and round(sg4.frr_test.min(), 3) == 0.002 and round(sg4.frr_test.max(), 3) == 0.004
               and (fst[(fst.dataset == D4) & (fst.calib == "boot") & fst.method.isin(["S3-SPRT", "P1-LLR"])].p_holm >= 0.05).all(),
               "sequential testing and likelihood-ratio fusion reached lower FRRs than the proposed design (0.001 against 0.002--0.004), though not significantly")
    check_true("v02: conclusion - proposed higher mean FRR than most serial rules on D4",
               sum(fv("D4", a, m) < fv("D4", a, "LR-P1-N2") for m in ["S1-Marcialis", "S2-Symmetric", "S4-Direct", "S3-SPRT"] for a in (1e-2, 1e-3, 1e-4)) > 6,
               "although the proposed design had a higher mean FRR than most of them")
    check_true("v02: conclusion - all serial rules except HYP below 0.005", ab4.max() < 0.005 and fsy[(fsy.dataset == D4) & (fsy.calib == "boot") & (fsy.method == "HYP")].frr_test.max() > 0.005,
               "the FRRs of all serial rules except the hyperbolic design stayed below 0.005")
    # compliance (Table 4, Sect. 6.4, Discussion, abstract)
    cc = lambda d, cal, ss: fco[(fco.dataset == DS.get(d, D4)) & (fco.calib == cal) & (fco.split_set == ss)]
    bD2 = cc("D2", "boot", "fresh"); check_true("v02: boot D2 19 of 20", (np.round(bD2.far_ok * 20, 6) == 19).all(), "met $\\alpha$ in 19 of 20 test halves on D2")
    bD1 = cc("D1", "boot", "fresh"); check_true(f"v02: boot D1 about 16 of 20 ({(bD1.far_ok * 20).round(2).tolist()})", (np.round(bD1.far_ok * 20) == 16).all(), "only in about 16 of 20 on D1")
    bD3 = cc("D3", "boot", "fresh"); check_true("v02: boot D3 14 to 17 of 20", sorted(np.round(bD3.far_ok * 20).astype(int)) == [14, 17, 17], "in 14 to 17 of 20 on D3")
    bD4 = cc("D4", "boot", "D4 0-9"); check("v02: boot D4 min of 10", (bD4.far_ok * 10).min(), "8.1", "and in 8.1 to 9.1 of 10 on D4"); check("v02: boot D4 max of 10", (bD4.far_ok * 10).max(), "9.1")
    xx = fco[(fco.calib == "xfit") & (fco.dataset != D4)]
    check_true(f"v02: xfit D1-D3 18 to 20 of 20 ({sorted((xx.far_ok * 20).round(2))})", np.round((xx.far_ok * 20).min()) == 18 and (xx.far_ok * 20).max() == 20, "they met $\\alpha$ in 18 to 20 of 20 test halves on D1--D3 and in all ten on D4")
    check_true("v02: xfit D4 all ten", (cc("D4", "xfit", "D4 0-9").far_ok == 1).all())
    from scipy.stats import norm as _ndist
    check("v02: expected compliance bootstrap %", _ndist.cdf(1.645 / np.sqrt(2)), "88", "below the 88\\% expected for a correct bound", scale=100)
    check("v02: expected compliance cross-fitted %", _ndist.cdf(1.645 * np.sqrt(2 / 3)), "91", "in line with the 91\\% expected under the same approximation", scale=100)
    b13 = pd.concat([bD1, bD3]).far_ok; xa = fco[fco.calib == "xfit"].far_ok
    check("v02: boot D1/D3 compliance min %", b13.min(), "70", "it met $\\alpha$ in 70\\%--85\\% of the test halves on D1 and D3, below the 88\\% expected", scale=100)
    check("v02: boot D1/D3 compliance max %", b13.max(), "85", scale=100)
    ball = fco[(fco.calib == "boot") & (fco.split_set != "original 0-9")].far_ok
    check_true(f"v02: abstract compliance 70-95 (boot, new splits and D4: {100 * ball.min():.1f}-{100 * ball.max():.1f}) / 90-100 held-out",
               round(100 * ball.min()) == 70 and round(100 * ball.max()) == 95 and round(100 * xa.min()) == 90 and xa.max() == 1,
               "Deployed proposed designs met the FAR requirement in 70\\%--95\\% of new test halves (88\\% expected) and in 90\\%--100\\% with held-out calibration")
    check_true("v02: boot below 88% on D1, D3 and D4 at 1e-3 only", (bD1.far_ok < 0.8776).all() and (bD3.far_ok < 0.8776).all() and (bD2.far_ok > 0.8776).all()
               and sorted(bD4[bD4.far_ok < 0.8776].alpha.round(6)) == [0.001], "below the 88\\% expected for a correct bound on D1 and D3 and at $\\alpha=10^{-3}$ on D4")
    check_true("v02: held-out 90-100 on D1-D3, all ten D4", round(100 * xa.min()) == 90 and fco[fco.calib == "xfit"].dataset.nunique() == 4,
               "whereas the held-out calibration met it in 90\\%--100\\% of the test halves of D1--D3 and in all ten D4 splits")
    fb = fco[(fco.split_set != "original 0-9")].groupby("calib").far_over_alpha.agg(["min", "max"])
    check("v02: held-out FAR/alpha min", fb.loc["xfit", "min"], "0.42", "(mean test FAR 0.42$\\alpha$--0.85$\\alpha$, against 0.73$\\alpha$--0.92$\\alpha$)")
    check("v02: held-out FAR/alpha max", fb.loc["xfit", "max"], "0.85"); check("v02: boot FAR/alpha min", fb.loc["boot", "min"], "0.73"); check("v02: boot FAR/alpha max", fb.loc["boot", "max"], "0.92")
    bl = fsy[fsy.method.isin(["S3-SPRT", "P0-Parallel", "P1-LLR", "P2-LogReg"])].groupby("calib").far_ok.agg(["min", "max"])
    check("v02: SPRT/fusion compliance boot min %", bl.loc["boot", "min"], "60", "met $\\alpha$ in 60\\%--100\\% of the test halves with the bootstrap and in 70\\%--100\\% with the held-out calibration", scale=100)
    check("v02: SPRT/fusion compliance held-out min %", bl.loc["xfit", "min"], "70", scale=100)
    check_true("v02: SPRT/fusion compliance max 100% under both", (bl["max"] == 1).all())
    check_true("v02: conclusion - SPRT/fusion 70-100 with held-out", round(100 * bl.loc["xfit", "min"]) == 70, "(the SPRT and parallel fusion in 70\\%--100\\%)")
    xb = fco[fco.split_set != "original 0-9"].pivot_table(index=["dataset", "alpha"], columns="calib", values="frr_test"); xb["rel"] = xb.xfit / xb.boot - 1; xb["abs"] = xb.xfit - xb.boot
    rr = lambda d: xb.loc[DS.get(d, D4)]
    check("v02: xfit cost D2 min %", rr("D2").rel.min(), "1", "it raised the FRR of the proposed designs by 1\\%--2\\% on D2 and 3\\%--5\\% on D3", scale=100); check("v02: xfit cost D2 max %", rr("D2").rel.max(), "2", scale=100)
    check("v02: xfit cost D3 min %", rr("D3").rel.min(), "3", scale=100); check("v02: xfit cost D3 max %", rr("D3").rel.max(), "5", scale=100)
    check("v02: xfit cost D1 min %", rr("D1").rel.min(), "58", "but by 58\\%--96\\% on D1 (0.006--0.007)", scale=100); check("v02: xfit cost D1 max %", rr("D1").rel.max(), "96", scale=100)
    check("v02: xfit cost D1 abs min", rr("D1")["abs"].min(), "0.006"); check("v02: xfit cost D1 abs max", rr("D1")["abs"].max(), "0.007")
    check("v02: xfit cost D4 min %", rr("D4").rel.min(), "65", "and 65\\%--84\\% on D4 (at most 0.0035)", scale=100); check("v02: xfit cost D4 max %", rr("D4").rel.max(), "84", scale=100)
    check("v02: xfit cost D4 max abs", rr("D4")["abs"].max(), "0.0035")
    # D4 presentation attacks (Sect. 6.5)
    r4 = pd.read_csv(f"{T}/spoof_trait_raw_lfw_x_fing.csv").groupby("method")[["spoof_face", "spoof_right_index"]].mean()
    check("v02: D4 face spoof proposed", r4.loc["LR-P1-N2", "spoof_face"], "0.96", "a perfect face artifact was accepted in 0.96 of the attempts by the proposed designs")
    oth = r4.drop(["LR-P1-N2", "P0-Parallel"]).spoof_face
    check("v02: D4 face spoof others min", oth.min(), "0.90", "and in 0.90--0.98 by every other method except sum fusion"); check("v02: D4 face spoof others max", oth.max(), "0.98")
    check("v02: D4 sum fusion right-index spoof", r4.loc["P0-Parallel", "spoof_right_index"], "0.82", "sum fusion instead accepted 0.82 of the perfect right-index artifacts")
    check_true("v02: D4 sum fusion face spoof low", r4.loc["P0-Parallel", "spoof_face"] < 0.5)
    Dz = np.load("../data/lfw_x_fing.npz"); Gz = Dz["genuine"]
    zmed = {m: float(np.median((Dz["S_" + m][Gz] - Dz["S_" + m][~Gz].mean()) / Dz["S_" + m][~Gz].std())) for m in ["face_S", "li_V", "ri_V"]}
    check("v02: D4 face genuine z median", zmed["face_S"], "6.6", "with a median genuine score of 18--22 impostor standard deviations against 6.6 for the face")
    check("v02: D4 finger z min", min(zmed["li_V"], zmed["ri_V"]), "18"); check("v02: D4 finger z max", max(zmed["li_V"], zmed["ri_V"]), "22")
    # two machines (Sect. 5.4, Online Resource 1, Sect. S4)
    pv_ = pd.read_csv(f"{R}/E3fresh_provenance.csv"); both = pv_[pv_.checked_against_pc.notna()]
    check("v02: splits computed on both machines", len(both), "7", "Seven splits were computed on both machines (seed 10 of D1, seeds 10 and 11 of D2 and D3, and seeds 0 and 1 of D4)")
    check_true("v02: the seven are those named", sorted(both.file) == sorted(["fresh_fing_x_face_s10.csv", "fresh_fing_x_fing_s10.csv", "fresh_fing_x_fing_s11.csv",
               "fresh_face_x_face_s10.csv", "fresh_face_x_face_s11.csv", "fresh_lfw_x_fing_s0.csv", "fresh_lfw_x_fing_s1.csv"]))
    diff_ = both[~both.checked_against_pc.str.startswith("identical")]
    check_true("v02: one split differs in 3 of 1484 designs, test rates identical", len(diff_) == 1 and "3 of 1484" in diff_.checked_against_pc.iloc[0]
               and "identical in all designs: True" in diff_.checked_against_pc.iloc[0] and diff_.file.iloc[0].startswith("fresh_fing_x_face"),
               "except for the training FRR and the impostor stage counts of 3 of the 1484 designs of the D1 split")
    npc = int((pv_.machine.str.startswith("local")).sum())
    check_true(f"v02: {npc} files from the PC = 36 of 40 new D2/D3 splits + 8 D4 splits", npc == 44 and
               pv_[pv_.machine.str.startswith("local")].file.str.contains("lfw").sum() == 8, "Thirty-six of the 40 new splits of D2 and D3 and eight of the D4 splits ran on a 24-core PC")
    print("=" * 30, "IJIS v04: repositioning, MIQP rationale, declarations, MLP baseline")
    v2t = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E2v2/e2v2_*_N2.csv")], ignore_index=True)
    txd = v2t[(v2t.method == "LR-BB exact dual") & (v2t.obj == "P1")].groupby("dataset").time.mean()
    check("v04: exact-dual LR-BB time min (MIQP rationale)", txd.min(), "0.51", "(0.60--1.93~s against 0.51--1.46~s for the exact-dual LR-BB")
    check("v04: exact-dual LR-BB time max", txd.max(), "1.46")
    check_true("v04: framework positioning in intro, contribution 2 and conclusion",
               "safe by construction, certifiably optimal over an exponent grid, and open to further posynomial constraints" in _norm(SRC["s1_intro.tex"])
               and "the value of the exact dual lies in the analysis it enables" in _norm(SRC["s1_intro.tex"]),
               "its value lies in these properties rather than in a lower FRR")
    check_true("v04: first-person plural for the authors' earlier work", "authors' group" not in ALL and r"In earlier work~\cite{lin2026}, we formulated" in ALL
               and r"We have applied LR to security resource allocation~\cite{chen2025}" in ALL)
    mt4 = _norm(SRC["main.tex"]); esm4 = _norm(SRC["ESM_1.tex"]); url4 = "https://github.com/EdSun3941/grid-certified-serial-biometrics"
    check_true("v04: declarations filled (funding, competing interests, code URL, contributions)",
               "No funding was received for conducting this study." in mt4
               and "The authors have no competing interests to declare that are relevant to the content of this article." in mt4
               and url4 in mt4 and url4 in esm4 and "To be completed" not in ALL and "to be inserted" not in ALL and "[repository" not in ALL)
    check_true("v04: AI use stated in one sentence in Sect. 5.4, separate subsection removed",
               "Use of generative AI" not in ALL and "sec:ai" not in ALL,
               "A large language model (Claude, Anthropic) assisted in writing the experiment code and drafting the text; the authors verified the code, the results, and the text and take full responsibility for the content.")
    import run_mlp as _rm
    from data import MATCHERS as _MA
    check_true("v04: MLP architecture in text = code (2 x 16 ReLU, Adam, L2 1e-4)", _rm.HIDDEN == (16, 16) and "alpha=1e-4" in open("run_mlp.py").read()
               and 'activation="relu", solver="adam"' in open("run_mlp.py").read(),
               "(MLP; two hidden layers of 16 rectified linear units, Adam, $L_2$ penalty $10^{-4}$)")
    nsc = [len(_MA[d]) for d in ["fing_x_face", "fing_x_fing", "face_x_face", "lfw_x_fing"]]
    check_true(f"v04: two to four scores per claim ({nsc})", min(nsc) == 2 and max(nsc) == 4, "because each claim offers only two to four scores")
    pm = pd.read_csv(f"{R}/logs/mlp_pc/progress_mlp.csv", comment="#", header=None, names=["ds", "seed", "rc", "sec", "ok"])
    check_true(f"v04: all 100 MLP runs on the PC finished ({len(pm)})", len(pm) == 100 and (pm.rc == 0).all() and pm.ok.all()
               and len(glob.glob(f"{R}/E3mlp/mlp_*.csv")) == 100, "as did all MLP runs")
    ms = tab("T_mlp_stats"); ms["rel"] = -ms.mean_diff / ms.frr_other; my = tab("T_mlp_system")
    b23 = ms[ms.dataset.isin(["fing_x_fing", "face_x_face"]) & (ms.calib == "boot")]
    bm = b23[b23.other == "LR-P1-N2"]
    check("v04: MLP vs proposed, D2/D3 bootstrap, rel. min %", bm.rel.min(), "7", "by 7\\%--14\\% (significantly in all 12 settings)", scale=100)
    check("v04: ... rel. max %", bm.rel.max(), "14", scale=100)
    check_true("v04: ... 12 settings, all significant, MLP lower in every split", len(bm) == 12 and (bm.p_unadj < 0.05).all() and (bm.n_mlp_lower == bm.n).all(),
               "it had lower FRR than the proposed design in every split with the bootstrap calibration")
    x2 = ms[(ms.dataset == "fing_x_fing") & (ms.calib == "xfit") & (ms.other == "LR-P1-N2")]
    check("v04: MLP vs proposed, D2 held-out, rel. min %", x2.rel.min(), "11", "and on D2 with the held-out calibration by 11\\%--14\\%", scale=100)
    check("v04: ... rel. max %", x2.rel.max(), "14", scale=100)
    check_true("v04: ... all three D2 held-out settings significant", len(x2) == 3 and (x2.p_unadj < 0.05).all() and (x2.mean_diff < 0).all())
    bl = b23[b23.other == "P1-LLR"]
    check_true(f"v04: MLP within 0.0035 of LLR fusion in the 12 settings (max {bl.mean_diff.abs().max():.5f})",
               len(bl) == 12 and round(bl.mean_diff.abs().max(), 4) <= 0.0035, "its mean FRR was within 0.0035 of that of LLR fusion")
    sl = ms[(ms.other == "P1-LLR") & (ms.p_unadj < 0.05)]
    check_true("v04: only significant MLP-LLR difference: D2, original splits, alpha 1e-4", len(sl) == 1 and sl.dataset.iloc[0] == "fing_x_fing"
               and sl.split_set.iloc[0] == "original 0-9" and np.isclose(sl.alpha.iloc[0], 1e-4),
               "significantly different only on D2 at $\\alpha=10^{-4}$ of the original splits")
    check_true("v04: no significant MLP difference on D1 and D4", (ms[ms.dataset.isin(["fing_x_face", "lfw_x_fing"])].p_unadj >= 0.05).all(),
               "On D1 and D4 no difference involving the MLP was significant")
    d3x = ms[(ms.dataset == "face_x_face") & (ms.calib == "xfit")]
    check_true("v04: no significant held-out difference from the proposed design or LLR fusion on D3",
               (d3x[d3x.other.isin(["LR-P1-N2", "P1-LLR"])].p_unadj >= 0.05).all(),
               "and on D3 no held-out difference from the proposed design or from LLR fusion was significant")
    mm = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/E3mlp/mlp_face_x_face_s*.csv")], ignore_index=True)
    fx = mm[(mm.calib == "xfit") & np.isclose(mm.alpha, 1e-4)].set_index("seed").frr_test
    fs_ = tab("T_fresh_selected"); ll = fs_[(fs_.dataset == "face_x_face") & (fs_.calib == "xfit") & np.isclose(fs_.alpha, 1e-4) & (fs_.method == "P1-LLR")].set_index("seed").frr_test
    worst = fx.idxmax()
    check("v04: D3 held-out outlier split, MLP FRR", fx[worst], "0.51", "reached an FRR of 0.51 at $\\alpha=10^{-4}$, twice that of LLR fusion (0.25)")
    check("v04: ... LLR fusion FRR on that split", ll[worst], "0.25")
    check_true(f"v04: ... ratio {fx[worst] / ll[worst]:.2f} is about two, and the next-worst MLP split is below 0.30",
               1.9 <= fx[worst] / ll[worst] <= 2.1 and fx.drop(worst).max() < 0.30)
    check("v04: MLP FAR compliance min %", my.far_ok.min(), "70", "The MLP met $\\alpha$ in 70\\%--100\\% of the test halves", scale=100)
    check("v04: MLP FAR compliance max %", my.far_ok.max(), "100", scale=100)
    check_true("v04: MLP positioned in intro and discussion", "whether linear or learned by a small neural network" in ALL,
               "a small neural fusion reached about the FRR of the likelihood-ratio fusion")
    print("=" * 30, "IJIS format checks")
    ab_ = open(f"{P}/sections/s0_abstract.tex").read(); ab_ = ab_.split("\\begin{abstract}")[1].split("\\keywords")[0]
    nw = len(re.split(r"\s+|--", re.sub(r"\$[^$]*\$", "X", ab_).replace("~", " ").strip()))
    check_true(f"IJIS: abstract has {nw} words (150-250)", 150 <= nw <= 250)
    kw = open(f"{P}/sections/s0_abstract.tex").read().split("\\keywords{")[1].split("}")[0].count("\\and") + 1
    check_true(f"IJIS: {kw} keywords (4-6)", 4 <= kw <= 6)
    lab = lambda f: dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", open(f).read()))
    if os.path.exists(f"{P}/ESM_1.aux") and os.path.exists(f"{P}/main.aux"):
        esm, mn = lab(f"{P}/ESM_1.aux"), lab(f"{P}/main.aux")
        for tlab, txt in [("tab:s-calib", "(Online Resource~1, Table~S2)"), ("tab:s-decomp", "(interquartile ranges in Online Resource~1, Table~S6)"),
                          ("tab:s-abl", "(Online Resource~1, Table~S11)")]:
            num = re.search(r"Table~(S\d+)", txt).group(1)
            check_true(f"IJIS: {tlab} is {num} in ESM_1", esm.get(tlab) == num, txt)
        check_true("IJIS: ESM cites main Tables 6 and 7 correctly", mn.get("tab:spoof") == "6" and mn.get("tab:cost") == "7",
                   "the single-modality spoof analysis is Table~6 of the paper")
        check_true("IJIS: ESM cites main Table 3 (system) correctly", mn.get("tab:system") == "3", "Table~3 of the paper")
        check_true("IJIS: ESM cites main Sect. 3.5 / 6.4 and Fig. 3 correctly",
                   mn.get("sec:calib") == "3.5" and mn.get("sec:cons") == "6.4" and mn.get("fig:scal") == "3",
                   "(Sect.~3.5 of the paper)")
        check_true("IJIS: Fig. 1 artwork refers to the GP as (3)", mn.get("eq:gp") == "3")
        check_true("IJIS: ESM Sect. S1/S4 cited from main text exist", esm.get("sec:s-p2") == "S1" and esm.get("sec:s-repro") == "S4",
                   "(Online Resource~1, Sect.~S4)")
        check_true("v04: MLP table is S19 in ESM_1", esm.get("tab:s-mlp") == "S19", "(Online Resource~1, Table~S19)")
    else:
        check_true("IJIS: main.aux and ESM_1.aux present (compile first)", False)

print("=" * 30, "Ablations (Section VI-F)")
ab = tab("T_ablation_paired"); av = lambda v, a, c: ab[(ab.variant == v) & (ab.method == "LR-P1-N2") & np.isclose(ab.alpha, a)][c].iloc[0]
check("sampled-point dominance FRR", av("abl_sample", 1e-3, "frr_test_variant"), "0.0097", "(0.0097 against 0.0093 at $\\alpha=10^{-3}$)")
check("main FRR (uncalibrated)", av("abl_sample", 1e-3, "frr_test_main"), "0.0093")
check("FAR-range restriction FRR", av("abl_region", 1e-3, "frr_test_variant"), "0.0432", "raised the FRR from 0.0093 to 0.0432")

print("=" * 30, "Generated tables are current")
before = {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in glob.glob(f"{P}/tables/*.tex")}
subprocess.run([sys.executable, "make_main_tables.py"], capture_output=True, check=True, env={**os.environ, "PAPER": P})
after = {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in glob.glob(f"{P}/tables/*.tex")}
check_true("main-paper table bodies unchanged after regeneration", before == after)
s_before = hashlib.md5(open(f"{P}/supp_tables.tex", "rb").read()).hexdigest()
subprocess.run([sys.executable, "make_supp_tables.py"], capture_output=True, check=True, env={**os.environ, "PAPER": P})
check_true("supplementary tables unchanged after regeneration", s_before == hashlib.md5(open(f"{P}/supp_tables.tex", "rb").read()).hexdigest())

print(f"\n{n_ok} checks passed, {len(fails)} failed" + (": " + "; ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
