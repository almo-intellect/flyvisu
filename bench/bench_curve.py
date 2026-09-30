"""Learning curve on the REAL task: how much validation loss do we give up by stopping early?

Stage 0 measured ~15-28 CU per 250,000-step run against ~242 usable CU, so the design only fits
if training can stop far earlier. This trains the real arms (flyvis_seam) on real Sintel and
records validation loss against iteration.

  python bench_curve.py sintel              download + extract MPI-Sintel (resumable, verified)
  python bench_curve.py train A 20000       one arm, n_iters steps, frequent checkpoints
  python bench_curve.py both 20000          arms A and B concurrently on one GPU

Everything except n_iters and the checkpoint interval is flyvis's own Stage-2 configuration:
batch 4, 19 frames, dt 0.02, original split, augmentation on.
"""
import json, os, shutil, subprocess, sys, time, urllib.error, urllib.request, zipfile
from pathlib import Path

os.environ.setdefault("FLYVIS_ROOT_DIR", "/content/flyvis_data")
sys.path.insert(0, "/content")

SINTEL_URL = "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"
SINTEL_BYTES = 5_627_783_629
CHKPT_EVERY_EPOCH = int(os.environ.get("CURVE_CHKPT_EVERY", "25"))
OUT = Path("/content/curve")


def log(*a):
    print(*a, flush=True)


def get_sintel():
    import flyvis
    sd = Path(flyvis.sintel_dir)
    sd.mkdir(parents=True, exist_ok=True)
    zpath = sd / "MPI-Sintel-complete.zip"
    marker = sd / ".extract_complete"
    if marker.exists():
        log(f"[sintel] already extracted at {sd}")
        return sd
    have = zpath.stat().st_size if zpath.exists() else 0
    if have > SINTEL_BYTES:
        log(f"[sintel] oversize zip ({have:,}); deleting")
        zpath.unlink(); have = 0
    fails, deadline = 0, time.time() + 3 * 3600
    while have < SINTEL_BYTES:
        if fails >= 20 or time.time() > deadline:
            sys.exit(f"STOP: download gave up at {have:,} bytes after {fails} failures")
        start = have
        req = urllib.request.Request(SINTEL_URL, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                cr = r.headers.get("Content-Range", "") or ""
                ok = (r.status == 206 and cr.startswith(f"bytes {have}-")) or (have == 0 and r.status == 200)
                if not ok:
                    sys.exit(f"STOP: server ignored Range (HTTP {r.status}, {cr!r})")
                with open(zpath, "ab") as f:
                    t0, last = time.time(), have
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - t0 > 30:
                            mb = (have - last) / 2**20 / (time.time() - t0)
                            log(f"  {have/2**30:6.2f} / {SINTEL_BYTES/2**30:.2f} GiB ({mb:.1f} MiB/s)")
                            t0, last = time.time(), have
            fails = 0 if have > start else fails + 1
        except Exception as e:
            fails += 1
            log(f"  interrupted at {have/2**30:.2f} GiB: {e}; retry {fails}/20")
            time.sleep(5)
            have = zpath.stat().st_size if zpath.exists() else 0
    log("[sintel] extracting ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(sd)
        bad = [i.filename for i in z.infolist()
               if not i.is_dir() and (not (sd / i.filename).is_file()
                                      or (sd / i.filename).stat().st_size != i.file_size)]
    if bad:
        sys.exit(f"STOP: extraction incomplete, {len(bad)} members missing/short")
    marker.write_text(str(SINTEL_BYTES))
    log(f"[sintel] ready at {sd}")
    return sd


def build(arm, steps, chkpt_every_epoch=None, tag=None, delete_if_exists=True):
    from flyvis.solver import MultiTaskSolver
    from flyvis.utils.config_utils import get_default_config
    import flyvis.utils.config_utils as cu
    import flyvis_seam as fs
    st = fs.register_flyvis_adapters()
    assert st["registered"], st["error"]
    base = get_default_config(
        path=str(Path(cu.__file__).parent.parent / "config" / "solver.yaml"),
        overrides=["task_name=flow", f"ensemble_and_network_id=curve/{tag or arm}",
                   f"task.n_iters={steps}",
                   f"scheduler.chkpt_every_epoch={chkpt_every_epoch or CHKPT_EVERY_EPOCH}"])
    cfg = fs.apply_arm_to_solver_config(base, **fs.ARM_PRESETS[arm])
    return MultiTaskSolver(config=cfg, delete_if_exists=delete_if_exists)


class StopHere(Exception):
    """Raised from the checkpoint hook once the requested iteration is reached."""


def lr_overrides(lr):
    """Scale flyvis's stepwise schedule (start -> stop, 10x apart) to a new start rate.
    lr_pen is defined by interpolation from lr_net in scheduler.yaml, so it follows."""
    if lr is None:
        return []
    lr = float(lr)
    return [f"scheduler.lr_net.start={lr}", f"scheduler.lr_net.stop={lr / 10}",
            f"scheduler.lr_dec.start={lr}", f"scheduler.lr_dec.stop={lr / 10}"]


def build_stock(tag, n_iters, lr=None):
    """Unmodified flyvis: PPNeuronIGRSynapses, clamp non_negative, no seam. The control."""
    from flyvis.solver import MultiTaskSolver
    from flyvis.utils.config_utils import get_default_config
    import flyvis.utils.config_utils as cu
    cfg = get_default_config(
        path=str(Path(cu.__file__).parent.parent / "config" / "solver.yaml"),
        overrides=["task_name=flow", f"ensemble_and_network_id=curve/{tag}",
                   f"task.n_iters={n_iters}", f"scheduler.chkpt_every_epoch={CHKPT_EVERY_EPOCH}"]
                  + lr_overrides(lr))
    return MultiTaskSolver(config=cfg, delete_if_exists=True)


def mode_stock(tag, n_iters, stop_iter, lr=None):
    import numpy as np
    solver = build_stock(tag, n_iters, lr)
    sch = solver.config.scheduler
    log(f"[train] {tag}: STOCK flyvis, dynamics {type(solver.network.dynamics).__name__}, "
        f"free params {solver.network.num_parameters.free}, n_iters {n_iters} (schedule), stop at {stop_iter}, "
        f"lr_net {sch.lr_net.start}->{sch.lr_net.stop} lr_dec {sch.lr_dec.start}->{sch.lr_dec.stop} "
        f"lr_pen {sch.lr_pen.start}->{sch.lr_pen.stop}")
    t0 = time.time()
    real_checkpoint = solver.checkpoint

    def checkpoint_and_report():
        real_checkpoint()
        try:
            v = float(np.asarray(solver.dir.validation.loss[:])[-1])
            log(f"LIVE arm={tag} iter={solver.iteration} val={v:.4f} elapsed_min={(time.time() - t0) / 60:.1f}")
        except Exception as e:
            log(f"LIVE arm={tag} iter={solver.iteration} (no val read: {e})")
        if solver.iteration >= stop_iter:
            raise StopHere()

    solver.checkpoint = checkpoint_and_report
    try:
        solver.train(overfit=False, initial_checkpoint=True)
    except StopHere:
        log(f"[train] {tag}: stopped at iteration {solver.iteration} as requested")
    curve_from(solver, tag, time.time() - t0)


def run_stock_jobs(jobs, stop_iter, label):
    """jobs: {tag: (n_iters, lr or None)}; all launched at once on one GPU, output relayed live."""
    import threading
    get_sintel()
    log(f"[{label}] warming the render cache with one dataset build")
    from flyvis.datasets.sintel import MultiTaskSintel
    MultiTaskSintel(tasks=["flow"], boxfilter={"extent": 15, "kernel_size": 13})
    procs, threads = {}, []
    t0 = time.time()

    def relay(p):
        for line in p.stdout:
            line = line.rstrip()
            if line.startswith(("LIVE", "CURVE", "[train]", "STOP", "Traceback")) or "Error" in line:
                log(line)

    for tag, (n_iters, lr) in jobs.items():
        cmd = [sys.executable, "-u", __file__, "stock", tag, str(n_iters), str(stop_iter)] + ([str(lr)] if lr else [])
        procs[tag] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        log(f"[{label}] launched {tag} (pid {procs[tag].pid})")
        th = threading.Thread(target=relay, args=(procs[tag],), daemon=True)
        th.start(); threads.append(th)
    for th in threads:
        th.join()
    for tag, p in procs.items():
        p.wait()
        log(f"[{label}] {tag} exit {p.returncode}")
    log(f"[{label}] total wall {time.time() - t0:.0f} s")


def mode_lrsweep(stop_iter, rates):
    """Stock flyvis at several learning rates, flyvis's real 250k schedule shape, stopped early."""
    jobs = {f"stock_lr{r:g}": (250000, r) for r in rates}
    run_stock_jobs(jobs, stop_iter, "lrsweep")


def mode_control(stop_iter):
    """Two stock runs at once: the schedule our arms got (n_iters=20000) and flyvis's real one (250000)."""
    import threading
    get_sintel()
    log("[control] warming the render cache with one dataset build")
    from flyvis.datasets.sintel import MultiTaskSintel
    MultiTaskSintel(tasks=["flow"], boxfilter={"extent": 15, "kernel_size": 13})
    jobs = {"stock_sched20k": 20000, "stock_sched250k": 250000}
    procs, threads = {}, []
    t0 = time.time()

    def relay(p):
        for line in p.stdout:
            line = line.rstrip()
            if line.startswith(("LIVE", "CURVE", "[train]", "STOP", "Traceback")) or "Error" in line:
                log(line)

    for tag, n_iters in jobs.items():
        procs[tag] = subprocess.Popen([sys.executable, "-u", __file__, "stock", tag, str(n_iters), str(stop_iter)],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        log(f"[control] launched {tag} (pid {procs[tag].pid})")
        th = threading.Thread(target=relay, args=(procs[tag],), daemon=True)
        th.start(); threads.append(th)
    for th in threads:
        th.join()
    for tag, p in procs.items():
        p.wait()
        log(f"[control] {tag} exit {p.returncode}")
    log(f"[control] total wall {time.time() - t0:.0f} s")


def curve_from(solver, arm, wall_s):
    import numpy as np
    d = solver.dir
    def arr(node, default=None):
        try:
            return np.asarray(node[:]).tolist()
        except Exception:
            return default if default is not None else []
    val = arr(d.validation.loss)
    it = arr(d.chkpt_iter)
    train = arr(d.loss)
    rows = list(zip(it, val)) if len(it) == len(val) else [(i, v) for i, v in enumerate(val)]
    rec = {"arm": arm, "iterations": solver.iteration, "wall_s": wall_s,
           "s_per_step": wall_s / max(solver.iteration, 1),
           "checkpoints": [{"iter": int(i), "val": float(v)} for i, v in rows],
           "train_loss_tail": [float(x) for x in train[-50:]]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"curve_{arm}.json").write_text(json.dumps(rec, indent=1))
    log(f"CURVE arm={arm} iterations={rec['iterations']} wall_s={wall_s:.0f} s_per_step={rec['s_per_step']:.3f}")
    for r in rec["checkpoints"]:
        log(f"CURVEPOINT arm={arm} iter={r['iter']} val={r['val']:.4f}")
    if rec["checkpoints"]:
        best = min(rec["checkpoints"], key=lambda r: r["val"])
        log(f"CURVEBEST arm={arm} best_val={best['val']:.4f} at_iter={best['iter']} "
            f"last_val={rec['checkpoints'][-1]['val']:.4f}")
    return rec


def mode_train(arm, steps):
    import numpy as np
    solver = build(arm, steps)
    log(f"[train] arm {arm}: free params {solver.network.num_parameters.free}, "
        f"{len(solver.task.train_data)} batches/epoch, checkpoint every {CHKPT_EVERY_EPOCH} epochs")
    t0 = time.time()
    real_checkpoint = solver.checkpoint

    def checkpoint_and_report():          # live progress: one line per validation pass
        real_checkpoint()
        try:
            v = float(np.asarray(solver.dir.validation.loss[:])[-1])
            log(f"LIVE arm={arm} iter={solver.iteration} val={v:.4f} elapsed_min={(time.time() - t0) / 60:.1f}")
        except Exception as e:
            log(f"LIVE arm={arm} iter={solver.iteration} (no val read: {e})")

    solver.checkpoint = checkpoint_and_report
    solver.train(overfit=False, initial_checkpoint=True)
    curve_from(solver, arm, time.time() - t0)


def mode_both(steps):
    import torch
    get_sintel()
    log("[both] warming the render cache with one dataset build")
    from flyvis.datasets.sintel import MultiTaskSintel
    MultiTaskSintel(tasks=["flow"], boxfilter={"extent": 15, "kernel_size": 13})
    import threading
    procs, threads = {}, []
    t0 = time.time()

    def relay(arm, p):                    # stream each child's progress lines as they arrive
        for line in p.stdout:
            line = line.rstrip()
            if line.startswith(("LIVE", "CURVE", "[train]", "STOP", "Traceback")) or "Error" in line:
                log(line)

    for arm in ("A", "B"):
        procs[arm] = subprocess.Popen([sys.executable, "-u", __file__, "train", arm, str(steps)],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        log(f"[both] launched arm {arm} (pid {procs[arm].pid})")
        th = threading.Thread(target=relay, args=(arm, procs[arm]), daemon=True)
        th.start(); threads.append(th)
    for th in threads:
        th.join()
    for arm, p in procs.items():
        p.wait()
        log(f"[both] arm {arm} exit {p.returncode}")
    log(f"[both] total wall {time.time() - t0:.0f} s")


# ------------------------------------------------------------ long run, checkpoints mirrored to Drive
DRIVE_FOLDER_NAME = "19 - FruitFly"
LONG_CHKPT_EVERY_EPOCH = 84          # 12 batches per epoch -> a checkpoint every 1,008 iterations


def find_drive_work():
    """The project folder on the mounted Drive (Shared Drive first), as the Stage 0 notebook resolves it."""
    roots = [Path("/content/drive/Shareddrives"), Path("/content/drive/MyDrive"),
             Path("/content/drive/Shared drives")]
    for r in roots:
        if not r.exists():
            continue
        if (r / DRIVE_FOLDER_NAME).is_dir():
            return r / DRIVE_FOLDER_NAME
        for sub in sorted(r.iterdir()):
            if sub.is_dir() and (sub / DRIVE_FOLDER_NAME).is_dir():
                return sub / DRIVE_FOLDER_NAME
    sys.exit(f"STOP: Drive is not mounted or has no folder named {DRIVE_FOLDER_NAME!r}; mount Drive first")


class DriveSync:
    """Mirror a results directory into Drive, copying only files whose size or mtime changed since
    the last sync in this process (a manifest, because Drive's FUSE mount does not keep mtimes).
    Each file is written under a temporary name and renamed, so a disconnect never leaves a
    half-written checkpoint under the real name."""

    def __init__(self, src, dst):
        self.src, self.dst, self.seen = Path(src), Path(dst), {}

    def run(self):
        n = 0
        for p in sorted(self.src.rglob("*")):
            if not p.is_file() or "__cache__" in p.parts:
                continue
            key = str(p.relative_to(self.src))
            s = p.stat()
            sig = (s.st_size, s.st_mtime_ns)
            if self.seen.get(key) == sig:
                continue
            q = self.dst / key
            q.parent.mkdir(parents=True, exist_ok=True)
            tmp = q.with_name(q.name + ".part")
            shutil.copyfile(p, tmp)
            os.replace(tmp, q)
            self.seen[key] = sig
            n += 1
        return n


def mode_longarm(arm, stop_iter, drive_run_dir, resume):
    """One arm on flyvis's real 250k schedule, stopped at stop_iter, a checkpoint every ~1,000
    iterations, every checkpoint mirrored to Drive. --resume continues from the Drive copy
    (network, decoder, optimizer and penalty states; the data-order RNG is not restored)."""
    import numpy as np, torch, flyvis
    from flyvis.utils.chkpt_utils import (resolve_checkpoints, recover_network, recover_decoder,
                                          recover_optimizer, recover_penalty_optimizers)
    tag = f"long_{arm}"
    drive_dst = Path(drive_run_dir) / tag
    results_dir = Path(flyvis.results_dir) / "flow" / "curve" / tag
    if resume:
        if not (drive_dst / "chkpts").is_dir():
            sys.exit(f"STOP: nothing to resume at {drive_dst}")
        if results_dir.exists():
            shutil.rmtree(results_dir)
        shutil.copytree(drive_dst, results_dir)
        log(f"[long] {arm}: restored {results_dir} from Drive")
    solver = build(arm, 250000, chkpt_every_epoch=LONG_CHKPT_EVERY_EPOCH, tag=tag, delete_if_exists=not resume)
    per_epoch = len(solver.task.train_data)
    if per_epoch != 12:
        log(f"[long] {arm}: NOTE {per_epoch} batches/epoch, checkpoints every {LONG_CHKPT_EVERY_EPOCH * per_epoch} iterations")
    if resume:
        ck = resolve_checkpoints(solver.dir)
        idx, path = ck.indices[-1], ck.paths[-1]
        state = torch.load(path, map_location=flyvis.device, weights_only=False)
        solver._last_chkpt_ind = solver._curr_chkpt_ind = idx
        solver.iteration = int(state["iteration"]) + 1      # stored value = last completed iteration
        solver.scheduler(solver.iteration)
        solver._val_loss = float(state.get("val_loss", float("inf")))
        recover_network(solver.network, state)
        recover_decoder(solver.decoder, state)
        recover_optimizer(solver.optimizer, state)
        recover_penalty_optimizers(solver.penalty.optimizers, state)
        try:      # train() rewrites loss.h5 with the new segment only; keep the earlier record
            seg = {"loss": np.asarray(solver.dir.loss[:]).tolist(),
                   "activity": np.asarray(solver.dir.activity[:]).tolist()}
            (solver.dir.path / f"loss_segment_to_iter_{solver.iteration - 1}.json").write_text(json.dumps(seg))
        except Exception as e:
            log(f"[long] {arm}: could not snapshot the earlier loss record: {e}")
        log(f"[long] {arm}: resumed from checkpoint {idx} ({Path(path).name}) at iteration {solver.iteration}")
    sch = solver.config.scheduler
    log(f"[long] {arm}: free params {solver.network.num_parameters.free}, {per_epoch} batches/epoch, "
        f"checkpoint every {sch.chkpt_every_epoch} epochs, schedule n_iters 250000, stop at {stop_iter}, "
        f"lr_net {sch.lr_net.start}->{sch.lr_net.stop}, Drive {drive_dst}")
    t0 = time.time()
    sync = DriveSync(solver.dir.path, drive_dst)
    real_checkpoint = solver.checkpoint

    def checkpoint_and_report():
        real_checkpoint()
        try:
            v = float(np.asarray(solver.dir.validation.loss[:])[-1])
            log(f"LIVE arm={arm} iter={solver.iteration} val={v:.4f} elapsed_min={(time.time() - t0) / 60:.1f}")
        except Exception as e:
            log(f"LIVE arm={arm} iter={solver.iteration} (no val read: {e})")
        try:
            n = sync.run()
            log(f"SYNC arm={arm} iter={solver.iteration} files={n}")
            status = {"arm": arm, "iteration": solver.iteration, "stop_iter": stop_iter,
                      "n_checkpoints": len(solver.checkpoints), "elapsed_min": (time.time() - t0) / 60,
                      "time": time.ctime(), "val": float(np.asarray(solver.dir.validation.loss[:])[-1])}
            (drive_dst.parent / f"status_{arm}.json").write_text(json.dumps(status))
        except Exception as e:
            log(f"SYNCFAIL arm={arm} iter={solver.iteration} {e}")
        if solver.iteration >= stop_iter:
            raise StopHere()

    solver.checkpoint = checkpoint_and_report
    try:
        solver.train(overfit=False, initial_checkpoint=not resume)
    except StopHere:
        log(f"[long] {arm}: stopped at iteration {solver.iteration} as requested")
    rec = curve_from(solver, tag, time.time() - t0)
    drive_dst.mkdir(parents=True, exist_ok=True)
    (drive_dst / "curve.json").write_text(json.dumps(rec, indent=1))
    sync.run()
    (drive_dst.parent / f"status_{arm}.json").write_text(json.dumps(
        {"arm": arm, "iteration": solver.iteration, "done": True, "time": time.ctime(),
         "elapsed_min": (time.time() - t0) / 60}))
    log(f"[long] {arm}: final sync done -> {drive_dst}")


def mode_long(stop_iter, run_name, resume):
    """Both arms at once with checkpoints mirrored to Drive, then M2 on every checkpoint of each arm.

      python bench_curve.py long 60000 long60k_2026-09-19            fresh run
      python bench_curve.py long 60000 long60k_2026-09-19 --resume   continue from the Drive copy
    """
    import threading
    work = find_drive_work()
    drive_run_dir = work / "runs" / run_name
    drive_run_dir.mkdir(parents=True, exist_ok=True)
    log(f"[long] Drive run dir: {drive_run_dir}")
    get_sintel()
    log("[long] warming the render cache with one dataset build")
    from flyvis.datasets.sintel import MultiTaskSintel
    MultiTaskSintel(tasks=["flow"], boxfilter={"extent": 15, "kernel_size": 13})
    procs, threads = {}, []
    t0 = time.time()

    def relay(p):
        for line in p.stdout:
            line = line.rstrip()
            if line.startswith(("LIVE", "SYNC", "CURVE", "[long]", "[train]", "STOP", "Traceback")) or "Error" in line:
                log(line)

    for arm in ("A", "B"):
        cmd = [sys.executable, "-u", __file__, "longarm", arm, str(stop_iter), str(drive_run_dir)]
        if resume:
            cmd.append("--resume")
        procs[arm] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        log(f"[long] launched arm {arm} (pid {procs[arm].pid})")
        th = threading.Thread(target=relay, args=(procs[arm],), daemon=True)
        th.start(); threads.append(th)
    for th in threads:
        th.join()
    for arm, p in procs.items():
        p.wait()
        log(f"[long] arm {arm} exit {p.returncode}")
    log(f"[long] training wall {time.time() - t0:.0f} s")
    m2 = Path(__file__).with_name("m2_released.py")
    for arm in ("A", "B"):
        out = drive_run_dir / f"m2_long_{arm}.json"
        log(f"[long] M2 on every checkpoint of arm {arm} -> {out}")
        r = subprocess.run([sys.executable, "-u", str(m2), "checkpoints", str(out), f"flow/curve/long_{arm}"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in r.stdout.splitlines():
            if line.startswith(("[m2]", "M2CURVE", "Traceback")) or "Error" in line:
                log(line)
        log(f"[long] M2 arm {arm} exit {r.returncode}")
        (drive_run_dir / f"m2_long_{arm}.log").write_text(r.stdout[-200000:])
    (drive_run_dir / "status_run.json").write_text(json.dumps(
        {"done": True, "time": time.ctime(), "training_wall_s": time.time() - t0}))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode == "sintel":
        get_sintel()
    elif mode == "long":
        mode_long(int(sys.argv[2]), sys.argv[3], "--resume" in sys.argv)
    elif mode == "longarm":
        mode_longarm(sys.argv[2], int(sys.argv[3]), sys.argv[4], "--resume" in sys.argv)
    elif mode == "train":
        mode_train(sys.argv[2], int(sys.argv[3]))
    elif mode == "stock":
        mode_stock(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]) if len(sys.argv) > 5 else None)
    elif mode == "control":
        mode_control(int(sys.argv[2]) if len(sys.argv) > 2 else 10500)
    elif mode == "lrsweep":
        mode_lrsweep(int(sys.argv[2]) if len(sys.argv) > 2 else 10500,
                     [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "2.5e-4,5e-4,1e-3").split(",")])
    else:
        mode_both(int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
    log("CURVE DONE")
