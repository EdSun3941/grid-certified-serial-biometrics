"""Parse NIST BSSR1 similarity files into dense score matrices (subject-aligned).

For each subset, rows = probe (user) samples, columns = enrollee samples.
All matchers of a subset are aligned to the same row/column subject order so that
row i / column j refer to the same probe/enrollee subject for every matcher.
Output: data/processed/bssr1_<subset>.npz with
  S_<matcher>  float32 [n_rows, n_cols]  similarity scores (larger = more similar)
  genuine      bool    [n_rows, n_cols]
  row_subject, col_subject : subject ids
Usage: python prepare_bssr1.py <bssr1_root> <out_dir> <subset> [...]
"""
import os, re, sys, time
import numpy as np

HEADER = "biometricscores.nist.gov"
# subset -> list of (matcher key, sets dir, sims dir)
SUBSETS = {
    "fing_x_face": [("face_C", "sets/dos", "sims/dos/face/C"), ("face_G", "sets/dos", "sims/dos/face/G"),
                    ("li_V", "sets/dos", "sims/dos/li/V"), ("ri_V", "sets/dos", "sims/dos/ri/V")],
    "face_x_face": [("face_C", "sets/dos", "sims/dos/face/C"), ("face_G", "sets/dos", "sims/dos/face/G")],
    "fing_x_fing": [("li_V", "sets/dos/li", "sims/dos/li/V"), ("ri_V", "sets/dos/ri", "sims/dos/ri/V")],
}
PAT = re.compile(r'<signature name="([^"]+)" subject_id="([^"]+)"')

def read_set(xml_path):
    names, sids = [], []
    with open(xml_path, encoding="utf-8") as f:
        for line in f:
            m = PAT.search(line)
            if m:
                names.append(m.group(1)); sids.append(m.group(2))
    return names, np.array(sids)

def read_sim(path, n_expected):
    with open(path) as f:
        lines = f.read().split("\n")
    assert lines[0].startswith(HEADER), path
    n = int(lines[1]); assert n == n_expected, (path, n, n_expected)
    return np.array(lines[2:2 + n], dtype=np.float64)

def parse_subset(root, subset, out_dir):
    base = os.path.join(root, subset)
    ref_rows = ref_cols = None
    out = {}
    for key, setdir, simdir in SUBSETS[subset]:
        t0 = time.time()
        u_names, u_sid = read_set(os.path.join(base, setdir, "users.xml"))
        e_names, e_sid = read_set(os.path.join(base, setdir, "enrollees.xml"))
        S = np.empty((len(u_names), len(e_names)), dtype=np.float32)
        for i, nm in enumerate(u_names):
            S[i] = read_sim(os.path.join(base, simdir, nm), len(e_names))
        if ref_rows is None:
            ref_rows, ref_cols = u_sid, e_sid
        else:  # align to reference subject order
            if not (np.array_equal(u_sid, ref_rows) and np.array_equal(e_sid, ref_cols)):
                ri = {s: k for k, s in enumerate(u_sid)}; ci = {s: k for k, s in enumerate(e_sid)}
                assert len(set(u_sid)) == len(u_sid) and len(set(e_sid)) == len(e_sid), "non-unique ids; cannot align"
                S = S[np.array([ri[s] for s in ref_rows])][:, np.array([ci[s] for s in ref_cols])]
                print("  aligned", key, "to reference order", flush=True)
        out["S_" + key] = S
        print(subset, key, S.shape, "min %.4g max %.4g" % (S.min(), S.max()), "%.1fs" % (time.time() - t0), flush=True)
    G = ref_rows[:, None] == ref_cols[None, :]
    out.update(genuine=G, row_subject=ref_rows, col_subject=ref_cols)
    print(subset, "genuine", int(G.sum()), "impostor", int((~G).sum()), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "bssr1_%s.npz" % subset), **out)
    print("saved", subset, flush=True)


# ---- chunked mode (each call parses a block of rows of one matcher; for time-limited shells)
def parse_chunk(root, subset, key, start, stop, out_dir):
    base = os.path.join(root, subset)
    spec = {k: (sd, sm) for k, sd, sm in SUBSETS[subset]}[key]
    u_names, u_sid = read_set(os.path.join(base, spec[0], "users.xml"))
    e_names, e_sid = read_set(os.path.join(base, spec[0], "enrollees.xml"))
    stop = min(stop, len(u_names))
    S = np.empty((stop - start, len(e_names)), dtype=np.float32)
    t0 = time.time()
    for i in range(start, stop):
        S[i - start] = read_sim(os.path.join(base, spec[1], u_names[i]), len(e_names))
    tmp = os.path.join(out_dir, "tmp_%s" % subset); os.makedirs(tmp, exist_ok=True)
    np.save(os.path.join(tmp, "S_%s_%05d.npy" % (key, start)), S)
    np.save(os.path.join(tmp, "rows_%s.npy" % key), u_sid); np.save(os.path.join(tmp, "cols_%s.npy" % key), e_sid)
    print(subset, key, "rows %d-%d of %d" % (start, stop, len(u_names)), "%.1fs" % (time.time() - t0), flush=True)

def combine(subset, out_dir):
    import glob
    tmp = os.path.join(out_dir, "tmp_%s" % subset); out = {}; ref_r = ref_c = None
    for key, _, _ in SUBSETS[subset]:
        parts = sorted(glob.glob(os.path.join(tmp, "S_%s_*.npy" % key)))
        S = np.concatenate([np.load(f) for f in parts], axis=0)
        r = np.load(os.path.join(tmp, "rows_%s.npy" % key)); c = np.load(os.path.join(tmp, "cols_%s.npy" % key))
        assert S.shape == (len(r), len(c)), (S.shape, len(r), len(c))
        if ref_r is None: ref_r, ref_c = r, c
        elif not (np.array_equal(r, ref_r) and np.array_equal(c, ref_c)):
            assert len(set(r)) == len(r) and len(set(c)) == len(c), "non-unique ids"
            ri = {s: k for k, s in enumerate(r)}; ci = {s: k for k, s in enumerate(c)}
            S = S[np.array([ri[s] for s in ref_r])][:, np.array([ci[s] for s in ref_c])]
            print("aligned", key)
        out["S_" + key] = S
        print(key, S.shape, "min %.4g max %.4g" % (S.min(), S.max()))
    G = ref_r[:, None] == ref_c[None, :]
    out.update(genuine=G, row_subject=ref_r, col_subject=ref_c)
    np.savez(os.path.join(out_dir, "bssr1_%s.npz" % subset), **out)
    print(subset, "genuine", int(G.sum()), "impostor", int((~G).sum()), "saved", flush=True)

if __name__ == "__main__":
    # python prepare_bssr1.py <root> <out> full <subset...> | chunk <subset> <key> <start> <stop> | combine <subset>
    root, out_dir, cmd = sys.argv[1], sys.argv[2], sys.argv[3]
    if cmd == "full":
        for s in sys.argv[4:]: parse_subset(root, s, out_dir)
    elif cmd == "chunk":
        parse_chunk(root, sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), out_dir)
    elif cmd == "combine":
        combine(sys.argv[4], out_dir)
