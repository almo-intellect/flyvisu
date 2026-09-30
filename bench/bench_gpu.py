"""Where does a flyvis training step actually spend its time, and what makes it cheaper?

Runs the REAL arms (flyvis_seam) at extent 15 through the REAL MultiTaskSolver, on mock Sintel
(step cost does not depend on image content, and this avoids the 5.24 GB download).

  python bench_gpu.py profile     one solver, ~12 steps under torch.profiler + nvidia-smi sampling
  python bench_gpu.py parallel    throughput with 1, 2, 4 concurrent solver processes
  python bench_gpu.py compile     19-frame rollout: eager vs torch.compile
  python bench_gpu.py all         all three, in that order

Prints one RESULT line per measurement so the numbers survive a screenshot.
"""
import os, sys, time, json, subprocess, threading, tempfile, shutil
from pathlib import Path

HERE = Path("/content")
os.environ.setdefault("FLYVIS_ROOT_DIR", "/content/flyvis_data")
sys.path.insert(0, str(HERE))
MOCK = Path(os.environ["FLYVIS_ROOT_DIR"]) / "mock_sintel"
ARM = os.environ.get("BENCH_ARM", "B")          # arm B = Dale + CfC, the costlier arm
STEPS = int(os.environ.get("BENCH_STEPS", "15"))


def log(*a):
    print(*a, flush=True)


def make_mock_sintel():
    """4 sequences x 20 frames of noise, in MPI-Sintel layout."""
    if (MOCK / ".done").exists():
        return MOCK
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    H, W = 436, 1024
    for seq in ["alley_1", "alley_2", "ambush_2", "bamboo_1"]:
        for sub in ["final", "flow"]:
            (MOCK / "training" / sub / seq).mkdir(parents=True, exist_ok=True)
        for i in range(20):
            Image.fromarray((rng.uniform(0, 1, (H, W)) * 255).astype(np.uint8)).save(
                MOCK / f"training/final/{seq}/frame_{i:04d}.png")
            with open(MOCK / f"training/flow/{seq}/frame_{i:04d}.flo", "wb") as f:
                np.array([202021.25], dtype=np.float32).tofile(f)
                np.array([W, H], dtype=np.int32).tofile(f)
                rng.standard_normal((H, W, 2)).astype(np.float32).tofile(f)
    (MOCK / ".done").write_text("ok")
    return MOCK


def build_solver(tag, n_iters):
    from flyvis.solver import MultiTaskSolver
    from flyvis.utils.config_utils import get_default_config
    import flyvis.utils.config_utils as cu
    import flyvis_seam as fs
    st = fs.register_flyvis_adapters()
    assert st["registered"], st["error"]
    base = get_default_config(
        path=str(Path(cu.__file__).parent.parent / "config" / "solver.yaml"),
        overrides=["task_name=flow", f"ensemble_and_network_id=bench/{tag}", f"task.n_iters={n_iters}",
                   f"+task.dataset.sintel_path={MOCK.as_posix()}", "task.original_split=false",
                   "scheduler.chkpt_every_epoch=100000"])
    cfg = fs.apply_arm_to_solver_config(base, **fs.ARM_PRESETS[ARM])
    solver = MultiTaskSolver(config=cfg, delete_if_exists=True)
    solver.checkpoint = lambda: None          # a checkpoint is a full validation pass; not part of a step
    n_batches = len(solver.task.train_data)
    return solver, n_batches


class GpuSampler(threading.Thread):
    """nvidia-smi utilization while something runs."""
    def __init__(self, period=0.2):
        super().__init__(daemon=True)
        self.period, self.stop_flag, self.util, self.mem = period, False, [], []

    def run(self):
        while not self.stop_flag:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
                u, m = out.splitlines()[0].split(",")
                self.util.append(float(u)); self.mem.append(float(m))
            except Exception:
                pass
            time.sleep(self.period)

    def stop(self):
        self.stop_flag = True
        self.join(timeout=3)
        n = max(len(self.util), 1)
        return (sum(self.util) / n, max(self.util or [0]), sum(self.mem) / n)


def mode_profile():
    import torch
    from torch.profiler import profile, ProfilerActivity
    solver, n_batches = build_solver("profile", 10)
    log(f"[setup] arm {ARM}: {n_batches} batches/epoch, batch_size {solver.config.task.batch_size}, "
        f"free params {solver.network.num_parameters.free}")
    solver.train(overfit=False, initial_checkpoint=False)      # warm up: caches, cudnn, rendering
    done_warm = solver.iteration
    solver.task.n_iters = done_warm + 12
    sampler = GpuSampler(); sampler.start()
    torch.cuda.synchronize(); t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        solver.train(overfit=False, initial_checkpoint=False)
    torch.cuda.synchronize(); wall = time.time() - t0
    util_mean, util_max, mem_util = sampler.stop()
    steps = solver.iteration - done_warm
    ka = prof.key_averages()
    def dev(e):
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(e, attr, None)
            if v:
                return v
        return 0.0
    gpu_us = sum(dev(e) for e in ka)
    launches = sum(e.count for e in ka if dev(e) > 0)
    log(f"RESULT profile steps={steps} wall_s={wall:.2f} per_step_s={wall/max(steps,1):.3f} "
        f"gpu_busy_s={gpu_us/1e6:.2f} gpu_busy_pct={100*gpu_us/1e6/wall:.1f} "
        f"kernel_launches={launches} per_step={launches//max(steps,1)} "
        f"smi_util_mean={util_mean:.0f} smi_util_max={util_max:.0f} smi_mem_util={mem_util:.0f} "
        f"peak_gb={torch.cuda.max_memory_allocated()/2**30:.2f}")
    log("[top CUDA kernels by total device time]")
    rows = sorted(ka, key=dev, reverse=True)[:8]
    for e in rows:
        log(f"   {dev(e)/1e3:9.1f} ms  {e.count:6d} x  {e.key[:58]}")
    log("[top CPU ops by self time]")
    for e in sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)[:6]:
        log(f"   {e.self_cpu_time_total/1e3:9.1f} ms  {e.count:6d} x  {e.key[:58]}")


def mode_steps():
    """One process: time STEPS solver iterations. Used by mode_parallel."""
    import torch
    solver, n_batches = build_solver(f"steps{os.getpid()}", 1)
    solver.train(overfit=False, initial_checkpoint=False)       # warm up
    done = solver.iteration
    solver.task.n_iters = done + STEPS
    torch.cuda.synchronize(); t0 = time.time()
    solver.train(overfit=False, initial_checkpoint=False)
    torch.cuda.synchronize(); wall = time.time() - t0
    steps = solver.iteration - done
    print(f"CHILD {os.getpid()} steps={steps} wall_s={wall:.2f} per_step_s={wall/max(steps,1):.3f} "
          f"peak_gb={torch.cuda.max_memory_allocated()/2**30:.2f}", flush=True)


def mode_parallel():
    import torch
    log(f"[parallel] warming the rendering cache once (arm {ARM})")
    make_mock_sintel()
    sampler = None
    for k in [int(x) for x in os.environ.get("BENCH_PROCS", "1,2,4").split(",")]:
        env = dict(os.environ, BENCH_ARM=ARM, BENCH_STEPS=str(STEPS))
        sampler = GpuSampler(); sampler.start()
        t0 = time.time()
        procs = [subprocess.Popen([sys.executable, __file__, "steps"], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(k)]
        outs = [p.communicate()[0] for p in procs]
        wall = time.time() - t0
        util_mean, util_max, _ = sampler.stop()
        per_step, peaks, steps_total = [], [], 0
        for o in outs:
            line = next((l for l in o.splitlines() if l.startswith("CHILD")), None)
            if line is None:
                log("   child failed:", o.strip().splitlines()[-3:])
                continue
            d = dict(kv.split("=") for kv in line.split()[2:])
            per_step.append(float(d["per_step_s"])); peaks.append(float(d["peak_gb"]))
            steps_total += int(d["steps"])
        if not per_step:
            log(f"RESULT parallel procs={k} FAILED"); continue
        slowest = max(per_step)
        log(f"RESULT parallel procs={k} steps_total={steps_total} wall_s={wall:.1f} "
            f"throughput_steps_per_s={steps_total/wall:.2f} per_step_s_worst={slowest:.3f} "
            f"per_step_s_mean={sum(per_step)/len(per_step):.3f} peak_gb_each={max(peaks):.2f} "
            f"smi_util_mean={util_mean:.0f} smi_util_max={util_max:.0f}")


def mode_compile():
    """The 19-frame rollout alone (no decoder/loss/dataloader): eager vs torch.compile."""
    import torch
    import flyvis_seam as fs
    from flyvis import Network
    fs.register_flyvis_adapters()
    net = Network(**fs.flyvis_network_kwargs(extent=15, **fs.ARM_PRESETS[ARM]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    B, T, dt = 4, 19, 0.02
    ss = net.steady_state(t_pre=0.5, dt=dt, batch_size=B, value=0.5)
    x = (net.stimulus.zero(B, T), net.stimulus.add_pre_stim(1.0), net.stimulus())[-1].clone()

    def run_once(fwd):
        opt.zero_grad()
        act = fwd(x, dt, state=ss)
        loss = act.pow(2).mean()
        loss.backward()
        opt.step()
        return float(loss.detach())

    def timeit(fwd, n, label):
        for _ in range(3):
            run_once(fwd)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            run_once(fwd)
        torch.cuda.synchronize()
        s = (time.time() - t0) / n
        log(f"RESULT compile {label} per_rollout_s={s:.3f} peak_gb={torch.cuda.max_memory_allocated()/2**30:.2f}")
        return s

    eager = timeit(net, 10, "eager")
    for mode in ("default", "reduce-overhead"):
        try:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            compiled = torch.compile(net, mode=mode, dynamic=False)
            run_once(compiled)                       # triggers tracing
            log(f"[compile:{mode}] first call (tracing) took {time.time()-t0:.0f} s")
            got = timeit(compiled, 10, f"torch.compile:{mode}")
            log(f"RESULT compile speedup mode={mode} x{eager/got:.2f}")
        except Exception as e:
            log(f"RESULT compile mode={mode} FAILED {type(e).__name__}: {str(e)[:200]}")
        finally:
            try:
                torch._dynamo.reset()
            except Exception:
                pass


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    make_mock_sintel()
    import torch
    log(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'} "
        f"cpus={os.cpu_count()} arm={ARM}")
    if mode in ("profile", "all"):
        mode_profile()
    if mode in ("parallel", "all"):
        mode_parallel()
    if mode in ("compile", "all"):
        mode_compile()
    if mode == "steps":
        mode_steps()
    log("BENCH DONE")
