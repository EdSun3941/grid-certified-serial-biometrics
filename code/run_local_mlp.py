"""Parallel launcher for the MLP baseline (run_mlp.py) on a multi-core PC: original splits 0-9 and new splits 10-29 of
D1-D3, and splits 0-9 of D4.  Single-threaded jobs; a job starts only while fewer than W jobs run and the available
physical memory and commit exceed a margin.  Usage: python run_local_mlp.py [W=12] [min free GB=3]"""
import os, subprocess, sys, time
from run_local2 import avail_gb                      # physical memory and (Windows) commit available, GB

W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MIN_FREE = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
MEM = {"fing_x_fing": 1.5, "face_x_face": 1.0, "lfw_x_fing": 0.5, "fing_x_face": 0.5}
JOBS = ([("fing_x_fing", s) for s in range(30)] + [("face_x_face", s) for s in range(30)]
        + [("fing_x_face", s) for s in range(30)] + [("lfw_x_fing", s) for s in range(10)])
LOG = "../results/logs_local"
out = lambda ds, s: f"../results/E3mlp/mlp_{ds}_s{s}.csv"


def main():
    os.makedirs(LOG, exist_ok=True); os.makedirs("../results/E3mlp", exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    todo = [j for j in JOBS if not os.path.exists(out(*j))]; running = {}
    prog = open(os.path.join(LOG, "progress_mlp.csv"), "a", buffering=1)
    prog.write(f"# start {time.strftime('%Y-%m-%d %H:%M:%S')} W={W} pending={len(todo)}\n")
    while todo or running:
        for j, (p, t0, fh) in list(running.items()):
            if p.poll() is not None:
                fh.close(); del running[j]; prog.write(f"{j[0]},{j[1]},{p.returncode},{time.time() - t0:.0f},{os.path.exists(out(*j))}\n")
        while todo and len(running) < W and avail_gb() > MIN_FREE + MEM[todo[0][0]]:
            j = todo.pop(0); fh = open(os.path.join(LOG, f"mlp_{j[0]}_s{j[1]}.log"), "w")
            running[j] = (subprocess.Popen([sys.executable, "run_mlp.py", j[0], str(j[1])], stdout=fh, stderr=subprocess.STDOUT, env=env), time.time(), fh)
            time.sleep(5)
        time.sleep(5)
    prog.write(f"# end {time.strftime('%Y-%m-%d %H:%M:%S')}\n"); prog.close()
    open(os.path.join(LOG, "DONE_MLP"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
