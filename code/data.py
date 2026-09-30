"""Data access, subject-disjoint splits, empirical ROC and fitting-point extraction."""
import os
import numpy as np

def data_file(ds):
    """Path of a processed BSSR1 subset (bssr1_<ds>.npz). Folder: $BSSR1_DIR, else ../data, else ../data/processed."""
    env = os.environ.get("BSSR1_DIR")
    for c in ([env] if env else ["../data", "../data/processed"]):
        for name in (f"bssr1_{ds}.npz", f"{ds}.npz"):          # {ds}.npz: non-BSSR1 subsets (e.g., lfw_x_fing, D4)
            p = os.path.join(c, name)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"bssr1_{ds}.npz not found; run prepare_bssr1.py or set BSSR1_DIR")

MATCHERS = {
    "fing_x_face": ["face_C", "face_G", "li_V", "ri_V"],
    "face_x_face": ["face_C", "face_G"],
    "fing_x_fing": ["li_V", "ri_V"],
    "lfw_x_fing": ["face_S", "li_V", "ri_V"],        # D4: SFace on LFW + BSSR1 fingers (chimeric), build_lfw_chimeric.py
}

def load_subset(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    return out

def split_subjects(row_subject, col_subject, seed, train_frac=0.5):
    """Subject-disjoint split. Returns (rows_tr, cols_tr, rows_te, cols_te) index arrays."""
    subj = np.unique(col_subject)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(subj))
    n_tr = int(round(train_frac * len(subj)))
    tr = set(subj[perm[:n_tr]].tolist())
    r_tr = np.array([s in tr for s in row_subject]); c_tr = np.array([s in tr for s in col_subject])
    return np.where(r_tr)[0], np.where(c_tr)[0], np.where(~r_tr)[0], np.where(~c_tr)[0]

def block(D, key, rows, cols):
    return D[key][np.ix_(rows, cols)]

def roc(gen, imp):
    """Similarity scores: accept iff s >= t. Thresholds = all distinct scores (+inf).
    Returns t (ascending), FAR(t)=P(imp>=t), FRR(t)=P(gen<t)."""
    t = np.unique(np.concatenate([gen, imp]))
    t = np.append(t, np.inf)
    imp_s, gen_s = np.sort(imp), np.sort(gen)
    far = 1.0 - np.searchsorted(imp_s, t, side="left") / len(imp_s)
    frr = np.searchsorted(gen_s, t, side="left") / len(gen_s)
    return t, far, frr

def staircase_points(far, frr, mode="corner"):
    """Operating staircase R(a) = min{FRR(t): FAR(t) <= a}, a right-continuous step function.
    corner: (x_{k+1}, y_k)  -- dominating these guarantees realized FRR <= g(a) (Prop. 1)
    sample: (x_k, y_k)"""
    o = np.lexsort((frr, far))  # FAR asc, then FRR asc
    x, y = far[o], frr[o]
    # For each FAR level keep the smallest FRR: threshold recovery picks the smallest
    # threshold t with FAR(t) <= a, which attains the minimum FRR at that FAR level.
    ux, idx = np.unique(x, return_index=True)
    ymin = np.minimum.reduceat(y, idx)
    x, y = ux, ymin
    if mode == "corner":
        xc, yc = x[1:], y[:-1]
    else:
        xc, yc = x, y
    m = (xc > 0) & (yc > 0)
    return xc[m], yc[m]

def thin(x, y, nbins=120, lo=None, hi=None):
    """Keep, in each of nbins equal-width log10(FAR) bins, the point with the largest FRR."""
    lo = np.log10(x.min()) if lo is None else np.log10(lo)
    hi = 0.0 if hi is None else np.log10(hi)
    b = np.clip(((np.log10(x) - lo) / max(hi - lo, 1e-12) * nbins).astype(int), 0, nbins - 1)
    keep = {}
    for i in range(len(x)):
        if b[i] not in keep or y[i] > y[keep[b[i]]]:
            keep[b[i]] = i
    k = np.array(sorted(keep.values()))
    return x[k], y[k]

def fitting_sets(gen, imp, mode="corner", region=None, nbins=120):
    """Returns (x_fit, y_fit) thinned set and (x_all, y_all) full dominance set within region."""
    _, far, frr = roc(gen, imp)
    x, y = staircase_points(far, frr, mode)
    if region is not None:
        m = (x >= region[0]) & (x <= region[1]); x, y = x[m], y[m]
    xf, yf = thin(x, y, nbins)
    return xf, yf, x, y
