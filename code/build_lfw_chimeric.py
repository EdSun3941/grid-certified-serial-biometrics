"""IJIS revision (reviewers 1 and 3): subset D4 with a contemporary face matcher.
Face: Labeled Faces in the Wild (LFW; lfw.tgz, SHA-256 055f7d9c...d536c0 as in scikit-learn), all 1680 identities
with at least two images; image 0001 is the reference and image 0002 the probe.  Faces are detected by YuNet
(OpenCV Zoo, face_detection_yunet_2023mar.onnx), aligned with the five landmarks, and compared by SFace
(OpenCV Zoo, face_recognition_sface_2021dec.onnx, 2021) with cosine similarity.  An image without a detection gets
the lowest score (-1) in every comparison, as BSSR1 does for failed comparisons.
Fingerprint: BSSR1 left and right index scores (matcher V) of 1680 randomly chosen D2 subjects (seed 20260930).
Each LFW identity is paired with one BSSR1 subject (chimeric subjects); face and fingerprint scores are therefore
independent across modalities by construction, while the two fingers keep their real dependence.
Output: ../data/lfw_x_fing.npz with S_face_S, S_li_V, S_ri_V, genuine, row_subject, col_subject, plus a log line.
Usage: python build_lfw_chimeric.py <lfw_src_dir>   (containing lfw/ and the two .onnx files)"""
import hashlib, json, os, sys, time
import cv2, numpy as np
from data import data_file

SRC = sys.argv[1] if len(sys.argv) > 1 else "../data/lfw_src"
DET = os.path.join(SRC, "face_detection_yunet_2023mar.onnx"); REC = os.path.join(SRC, "face_recognition_sface_2021dec.onnx")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    root = os.path.join(SRC, "lfw"); ids = sorted(i for i in os.listdir(root) if len(os.listdir(os.path.join(root, i))) >= 2)
    det = cv2.FaceDetectorYN.create(DET, "", (250, 250), 0.5, 0.3, 50)
    rec = cv2.FaceRecognizerSF.create(REC, "")
    feats = {"ref": [], "probe": []}; fails = {"ref": [], "probe": []}
    for k, pid in enumerate(ids):
        files = sorted(os.listdir(os.path.join(root, pid)))
        for role, fn in (("ref", files[0]), ("probe", files[1])):
            img = cv2.imread(os.path.join(root, pid, fn)); h, w = img.shape[:2]
            det.setInputSize((w, h)); _, faces = det.detect(img)
            if faces is None or len(faces) == 0:
                det.setScoreThreshold(0.2); _, faces = det.detect(img); det.setScoreThreshold(0.5)
            if faces is None or len(faces) == 0:
                feats[role].append(np.zeros(128, np.float32)); fails[role].append(k); continue
            c = np.array([w / 2, h / 2]); ctr = faces[:, 0:2] + faces[:, 2:4] / 2
            f = faces[int(np.argmin(np.linalg.norm(ctr - c, axis=1)))]          # the face nearest to the image centre
            al = rec.alignCrop(img, f); v = rec.feature(al).ravel().astype(np.float64)
            feats[role].append((v / np.linalg.norm(v)).astype(np.float32))
    R = np.stack(feats["ref"]); P = np.stack(feats["probe"])
    S_face = (P @ R.T).astype(np.float32)                                        # cosine similarity, probe x reference
    S_face[fails["probe"], :] = -1.0; S_face[:, fails["ref"]] = -1.0
    n = len(ids)
    # fingerprints: BSSR1 D2 subjects, random subset
    D = np.load(data_file("fing_x_fing")); assert (D["row_subject"] == D["col_subject"]).all()
    idx = np.sort(np.random.default_rng(20260930).choice(len(D["row_subject"]), n, replace=False))
    perm = np.random.default_rng(20260931).permutation(n)                       # LFW identity k <-> BSSR1 subject idx[perm[k]]
    sel = idx[perm]
    S_li = D["S_li_V"][np.ix_(sel, sel)]; S_ri = D["S_ri_V"][np.ix_(sel, sel)]
    assert D["genuine"][np.ix_(sel, sel)].diagonal().all()
    subj = np.array([f"L{k:04d}" for k in range(n)])
    out = os.path.join(os.path.dirname(data_file("fing_x_fing")), "lfw_x_fing.npz")
    np.savez_compressed(out, S_face_S=S_face, S_li_V=S_li, S_ri_V=S_ri, genuine=np.eye(n, dtype=bool), row_subject=subj, col_subject=subj,
                        lfw_identity=np.array(ids), bssr1_subject=D["row_subject"][sel])
    log = dict(n_subjects=n, face_detect_fail_ref=len(fails["ref"]), face_detect_fail_probe=len(fails["probe"]),
               sha256_lfw_tgz_expected="055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0",
               sha256_det=sha256(DET), sha256_rec=sha256(REC), opencv=cv2.__version__, seconds=round(time.time() - t0, 1),
               out=out, sha256_out=sha256(out))
    print(json.dumps(log, indent=1))
    json.dump(log, open(os.path.join(os.path.dirname(out), "lfw_x_fing_build.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
