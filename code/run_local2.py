"""Parallel launcher (second version) for the fresh-seed replication (D1-D3, splits 10-29) and subset D4 (splits 0-9).
Each job runs `python run_fresh.py <dataset> <seed>` single-threaded (BLAS threads = 1); results do not depend on
the number of workers.  A job is claimed by an exclusive lock file (../results/locks), so several launchers, or jobs
started by an earlier launcher (pre-locked with --lock), never duplicate work.  A new job starts only while fewer than
W run_fresh processes are running on the machine and both the available physical memory and the available commit
(Windows: GlobalMemoryStatusEx) exceed a margin.
Usage: python run_local2.py [W=22] [min free GB=2.5] [--lock ds:seed ...]
Logs: ../results/logs_local/<dataset>_s<seed>.log; progress: ../results/logs_local/progress2.csv; done flag: DONE2."""
import os, subprocess, sys, time
import psutil

args = [a for a in sys.argv[1:] if not a.startswith("--")]
W = int(args[0]) if len(args) > 0 else 22
MIN_FREE = float(args[1]) if len(args) > 1 else 2.5
PRELOCK = sys.argv[sys.argv.index("--lock") + 1:] if "--lock" in sys.argv else []
MEM = {"fing_x_fing": 1.5, "face_x_face": 1.0, "lfw_x_fing": 0.5, "fing_x_face": 0.5}   # expected peak commit GB per job
JOBS = ([("fing_x_fing", s) for s in range(10, 30)] + [("face_x_face", s) for s in range(10, 30)]
        + [("lfw_x_fing", s) for s in range(10)] + [("fing_x_face", s) for s in range(10, 30)])   # longest first
LOG, LOCK = "../results/logs_local", "../results/locks"


def out(ds, s):
    return f"../results/E3fresh/fresh_{ds}_s{s}.csv"


def claim(ds, s):
    try:
        os.close(os.open(os.path.join(LOCK, f"{ds}_s{s}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError:
        return False


def avail_gb():
    phys = psutil.virtual_memory().available
    if os.name == "nt":
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        phys = min(phys, m.ullAvailPageFile)                      # available commit
    return phys / 2 ** 30


def n_fresh():
    n = 0
    for p in psutil.process_iter(["cmdline"]):
        try:
            if any("run_fresh.py" in c for c in (p.info["cmdline"] or [])): n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return n


def main():
    for d in (LOG, LOCK, "../results/E3fresh"): os.makedirs(d, exist_ok=True)
    for x in PRELOCK:
        ds, s = x.split(":"); claim(ds, int(s))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    prog = open(os.path.join(LOG, "progress2.csv"), "a", buffering=1)
    prog.write(f"# start {time.strftime('%Y-%m-%d %H:%M:%S')} W={W} min_free={MIN_FREE} prelocked={PRELOCK}\n")
    running = {}
    while True:
        for j, (p, t0, fh) in list(running.items()):
            if p.poll() is not None:
                fh.close(); del running[j]
                prog.write(f"{j[0]},{j[1]},{p.returncode},{time.time() - t0:.0f},{os.path.exists(out(*j))}\n")
        todo = [j for j in JOBS if not os.path.exists(out(*j)) and not os.path.exists(os.path.join(LOCK, f"{j[0]}_s{j[1]}.lock"))]
        if not todo and not running: break
        free = avail_gb()
        for j in todo:
            if n_fresh() >= W or free < MIN_FREE + MEM[j[0]]: break
            if not claim(*j): continue
            fh = open(os.path.join(LOG, f"{j[0]}_s{j[1]}.log"), "w")
            p = subprocess.Popen([sys.executable, "run_fresh.py", j[0], str(j[1])], stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[j] = (p, time.time(), fh); free -= MEM[j[0]]
            time.sleep(20); free = min(free, avail_gb())
        time.sleep(15)
    prog.write(f"# end {time.strftime('%Y-%m-%d %H:%M:%S')}\n"); prog.close()
    open(os.path.join(LOG, "DONE2"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
