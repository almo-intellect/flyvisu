"""Generate 00_stage0_setup_and_benchmark.ipynb.

The notebook is generated rather than hand-edited so the JSON is valid by
construction. Round 1 hand-wrote 28 cells of notebook JSON and the reviewer had
to validate it separately; this removes that failure mode.

Run: python build_stage0_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# =============================================================================
md(
    r"""
# Stage 0 — environment, premise, feasibility, cost

**One job: produce the numbers that size the whole project.** Specifically:

| # | Question | Cell |
|---|---|---|
| a | Which GPU did we get, and what does it burn per hour? | §1 |
| b | Do the released-ensemble premise numbers reproduce? | §5 |
| c | Do *our* modules run against a real flyvis connectome? | §6 |
| d | What does ONE training run cost, and how many seeds can we afford? | §9 |

## Safety rules this notebook follows

- **Nothing expensive runs before the cheap checks pass.** Every compute-unit-consuming
  cell is marked `# $$$` and guarded *outside* any `try`.
- **Every network call has a timeout.** The MPI-Sintel archive is **5.24 GB over plain
  HTTP**; it is gated behind an explicit opt-in in §7 and is *not* needed for §1–§6.
- **Two trained arms, unconditionally.** The released 50-model ensemble records
  `seed: 0` for every model and models `000`/`007` are byte-identical, so it is a
  *reference distribution*, never a baseline arm.

## Runtime — pin it before you start

`Runtime > Change runtime type > Runtime version > ` **`2026.07`** (Python 3.12.13).

Colab's **"Latest" ships Python 3.13.15**, which flyvis does *not* support
(`requires-python = ">=3.9,<3.13"`) — verified on this notebook 2026-09-16. When you
later switch the accelerator to T4/L4 for §9, change **only** the accelerator; do not
let Runtime version revert to "Latest".

⚠️ **`Runtime > Disconnect and delete runtime` RESETS this pin back to "Latest."**
A plain `Runtime > Restart session` preserves it. Verified 2026-09-16.

⚠️ **After §3 installs flyvis you may need `Runtime > Restart session` once**, then Run
all again. An editable install writes a `.pth` that a *running* kernel has already
scanned, so §4 can fail with `ModuleNotFoundError: No module named 'flyvis'`. A restart
fixes it and §3 then skips the install.

**§1–§6 need no GPU.** Run them on a **CPU** runtime — measured at ~0.08 CU/hour, i.e.
effectively free — and they answer (a), (b) and (c) *on CPU*.

⚠️ **For §9, switch to T4 and run §1–§9 in ONE T4 session.** Changing the accelerator,
or a disconnect that recycles the VM, gives a NEW VM: everything under `/content` (the
flyvis install, pretrained models, the Sintel zip and extraction) is gone; only Drive
survives. The §9 gate also refuses a §6 pass recorded on CPU — CUDA-only bugs exist
(`import flyvis` makes CUDA the default device), so §6 must pass on the GPU that §9 bills.

> Run cells in order. If a cell raises `SystemExit`, stop and read the message —
> that is a deliberate gate, not a crash.
"""
)

# =============================================================================
md("## §1 — Environment probe (free)\n\nNo installs, no downloads. Fails fast on a bad runtime.")

code(
    r"""
import os, sys, subprocess, json, time, platform, shlex, re

# Compute-unit burn rates. Values drift; treat as approximate and re-check
# https://colab.research.google.com/signup for current numbers.
CU_PER_HOUR = {"T4": 1.76, "L4": 4.00, "V100": 4.90, "A100": 15.00, "CPU": 0.00}
BUDGET_CU = 289.97  # as shown on the Colab subscription page

def sh(cmd, timeout=120):
    # Never call subprocess without a timeout on a metered runtime.
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s: {cmd}"

def q(p):
    # ALWAYS wrap a path before interpolating it into sh(). Measured 2026-09-16: the
    # Drive working folder is ".../Almo-2000-Research&Development/19 - FruitFly", which
    # contains BOTH spaces and an ampersand. Unquoted, /bin/sh split it and backgrounded
    # part of the command -- "/bin/sh: 1: Development/19: not found" -- and §6's module
    # self-test failed for that reason alone.
    return shlex.quote(str(p))

env = {"python": platform.python_version(), "platform": platform.platform()}

rc, out = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", timeout=60)
gpu_name, gpu_mem = "CPU", "0"
if rc == 0 and out.strip():
    parts = out.strip().splitlines()[0].split(",")
    gpu_name = parts[0].strip()
    gpu_mem = parts[1].strip() if len(parts) > 1 else "?"

short = "CPU"
for k in ("A100", "L4", "T4", "V100"):
    if re.search(rf"\b{k}\b", gpu_name):      # whole token: "L40S" must not match "L4"
        short = k
        break
if gpu_name != "CPU" and short == "CPU":
    raise SystemExit(
        f"STOP: GPU {gpu_name!r} has no entry in CU_PER_HOUR, so cost per run would be computed\n"
        f"as 0 CU and §10 would print a bogus GO.\n"
        f"FIX: add its rate from Colab's Resources panel to CU_PER_HOUR and its name to the tuple "
        f"above, then re-run this cell."
    )
env["gpu"] = gpu_name
env["gpu_short"] = short
env["gpu_memory"] = gpu_mem
env["cu_per_hour"] = CU_PER_HOUR.get(short, 0.0)

print(f"Python   : {env['python']}")
print(f"Platform : {env['platform']}")
print(f"GPU      : {gpu_name}  ({gpu_mem})")
if env["cu_per_hour"] > 0:
    print(f"CU/hour  : {env['cu_per_hour']}   -> {BUDGET_CU/env['cu_per_hour']:.0f} h on {BUDGET_CU} CU")
else:
    print("CU/hour  : n/a - CPU runtime (measured ~0.08 CU/h, effectively free)")

# --- gates ---------------------------------------------------------------
pv = sys.version_info
if not ((3, 9) <= (pv.major, pv.minor) < (3, 13)):
    raise SystemExit(
        f"STOP: flyvis requires Python >=3.9,<3.13 (pyproject.toml at v1.2.0); this runtime "
        f"is {platform.python_version()}.\n\n"
        f"FIX: Runtime > Change runtime type > Runtime version > 2026.07 (ships Python 3.12.13).\n\n"
        f"Measured 2026-09-16: Colab's 'Latest' runtime ships Python 3.13.15, which flyvis does\n"
        f"NOT support. Pinning 2026.07 is required. NOTE that 'Disconnect and delete runtime'\n"
        f"RESETS this pin to 'Latest' - a plain 'Restart session' does not. When you later\n"
        f"switch the accelerator to T4/L4, change ONLY the accelerator."
    )
print("\n[OK] Python is inside flyvis's supported range.")

if short == "A100":
    print("\n" + "!" * 76)
    print("WARNING: this is an A100 at ~15 CU/hr — it will eat the 289.97 CU budget in ~19 h.")
    print("This workload is kernel-launch-latency bound on ~1.5M sparse edges and will NOT")
    print("run 6x faster than a T4 for 8x the cost. Runtime > Change runtime type > L4 or T4,")
    print("then re-run this cell. Continue only if you deliberately want the A100.")
    print("!" * 76)
elif short == "CPU":
    print("\nWARNING: no GPU detected. §9 timing will be meaningless. Runtime > Change runtime type.")
else:
    print(f"[OK] {short} is a sensible choice for this workload.")
"""
)

# =============================================================================
md(
    """
## §2 — Mount Drive and resolve the working folder (free)

The project folder lives in a **Shared Drive**, which Colab mounts under
`/content/drive/Shareddrives/...` — *not* `MyDrive`. This cell resolves it by
searching both roots and fails loudly rather than silently writing to the wrong place.
"""
)

code(
    r"""
from pathlib import Path

DRIVE_FOLDER_NAME = "19 - FruitFly"
DRIVE_FOLDER_ID = "1EMFHFcWOqn-w5Qq1i-fHVRUVzIMaazc5"  # for reference / manual checks

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
WORK = None

if IN_COLAB:
    from google.colab import drive as _gdrive
    _gdrive.mount("/content/drive", force_remount=False)  # idempotent

    roots = [Path("/content/drive/Shareddrives"), Path("/content/drive/MyDrive"),
             Path("/content/drive/Shared drives")]
    candidates = []
    for r in roots:
        if not r.exists():
            continue
        if (r / DRIVE_FOLDER_NAME).is_dir():
            candidates.append(r / DRIVE_FOLDER_NAME)
        for sub in sorted(r.iterdir()):          # one level down inside each shared drive
            if sub.is_dir() and (sub / DRIVE_FOLDER_NAME).is_dir():
                candidates.append(sub / DRIVE_FOLDER_NAME)

    if not candidates:
        searched = "\n  ".join(str(r) for r in roots)
        raise SystemExit(
            f"STOP: could not find a folder named {DRIVE_FOLDER_NAME!r} under:\n  {searched}\n"
            f"Open https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID} and confirm the "
            f"folder name, or set WORK manually in the next cell."
        )
    WORK = candidates[0]
    if len(candidates) > 1:
        print("NOTE: multiple matches; using the first:")
        for c in candidates:
            print("   ", c)
else:
    WORK = Path.cwd()
    print("Not on Colab — using the current directory as the working root.")

RESULTS = WORK / "stage0_results.json"
CKPT_DIR = WORK / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nWORK    : {WORK}")
print(f"writable: {os.access(WORK, os.W_OK)}")
if not os.access(WORK, os.W_OK):
    raise SystemExit(f"STOP: {WORK} is not writable.")

def load_results():
    if not RESULTS.exists():
        return {}
    txt = RESULTS.read_text()          # let OSError propagate: never turn it into {}
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        bak = RESULTS.with_name(f"stage0_results.corrupt-{int(time.time())}.json")
        bak.write_text(txt)
        raise SystemExit(f"STOP: {RESULTS} is not valid JSON ({e}). Copied to {bak}. Repair or "
                         "delete it, then re-run; refusing to overwrite the stored results.")

def save_results(**kw):
    d = load_results()
    d.update(kw)
    tmp = RESULTS.with_name(RESULTS.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, default=str))
    os.replace(tmp, RESULTS)           # old or new file, never half-written
    return d

save_results(env=env, work=str(WORK))
print(f"[OK] results file: {RESULTS}")

# flyvis resolves its data directories from FLYVIS_ROOT_DIR *at import time* and also calls
# torch.set_default_device() there. It must therefore be set before ANY import of flyvis --
# including the probe inside have_flyvis() in the next cell, which is why it lives here and
# not next to the real import in §4.
#
# Deliberately local Colab disk, NOT Drive: the Sintel archive alone is 5.24 GB, and writing
# it to a Shared Drive would be slow and would eat Drive quota (35 GB already in use).
# Only results and checkpoints go to Drive, written explicitly.
FLYVIS_DATA = Path("/content/flyvis_data") if IN_COLAB else (Path.cwd() / "flyvis_data")
FLYVIS_DATA.mkdir(parents=True, exist_ok=True)
os.environ["FLYVIS_ROOT_DIR"] = str(FLYVIS_DATA)
print(f"[OK] FLYVIS_ROOT_DIR = {os.environ['FLYVIS_ROOT_DIR']}  (set before any flyvis import)")
"""
)

# =============================================================================
md(
    """
## §3 — Install flyvis (free; a few minutes)

`pip install` and `git clone` both get explicit timeouts. Re-running this cell is safe.

If §4 then reports `ModuleNotFoundError: No module named 'flyvis'`, do
`Runtime > Restart session` (NOT "delete runtime") and Run all again — this cell will
skip the install the second time.
"""
)

code(
    r"""
FLYVIS_TAG = "v1.2.0"
FLYVIS_COMMIT = "92b3845cc426dd309a1a0e1b3890156c42e14021"

# Deliberately NOT "/content/flyvis". Colab's cwd is /content, so a directory named
# `flyvis` sitting there SHADOWS the installed package: `import flyvis` resolves to an
# empty PEP-420 namespace package and you get
#     ImportError: cannot import name 'Network' from 'flyvis' (unknown location)
# Measured 2026-09-16. The repo root must not share its name with the package it ships.
FLYVIS_SRC = Path("/content/flyvis_repo") if IN_COLAB else Path.cwd() / "flyvis_repo"

def have_flyvis():
    try:
        import flyvis  # noqa: F401
        return True
    except Exception:
        return False

if have_flyvis():
    print("[OK] flyvis already importable; skipping install.")
else:
    if not (FLYVIS_SRC / ".git").exists():
        print(f"cloning flyvis {FLYVIS_TAG} ...")
        rc, out = sh(
            f"git clone --branch {FLYVIS_TAG} --depth 1 "
            f"https://github.com/TuragaLab/flyvis.git {q(FLYVIS_SRC)}",
            timeout=600,
        )
        print(out[-2000:])
        if rc != 0:
            raise SystemExit(f"STOP: git clone failed (rc={rc}). See output above.")
    print("installing (this can take several minutes) ...")
    rc, out = sh(f"pip install -q -e {q(FLYVIS_SRC)}", timeout=2400)
    print(out[-3000:])
    if rc != 0:
        raise SystemExit(f"STOP: pip install failed (rc={rc}). See output above.")
    print("\n[OK] installed.")
    print("If §4 says \"No module named 'flyvis'\": Runtime > Restart session, then Run all.")
    print("(An editable install writes a .pth that a RUNNING kernel has already scanned.)")
"""
)

# =============================================================================
md("## §4 — Verify flyvis and the connectome (free)")

code(
    r"""
import torch

# FLYVIS_ROOT_DIR was set in §2, before any flyvis import. Assert that here rather than
# setting it, because setting it now would be silently too late.
if os.environ.get("FLYVIS_ROOT_DIR") != str(FLYVIS_DATA):
    raise SystemExit("STOP: FLYVIS_ROOT_DIR is not set as §2 left it. Re-run §2 first.")

try:
    import flyvis
except ModuleNotFoundError as e:
    raise SystemExit(
        f"STOP: {e}\n\n"
        "FIX: Runtime > Restart session (NOT 'delete runtime', which would also reset the\n"
        "2026.07 pin), then Run all again. §3 installed flyvis with `pip install -e`, which\n"
        "writes a .pth into site-packages that this ALREADY-RUNNING kernel has scanned past.\n"
        "A restart makes it importable and §3 will then skip the install."
    )

# Namespace-package guard. If a directory named `flyvis` is on sys.path ahead of the
# installed package, `import flyvis` silently yields an EMPTY namespace package whose
# __file__ is None, and the next line fails with an unhelpful
# "cannot import name 'Network' from 'flyvis' (unknown location)". Catch it here with a
# message that says what to do. See the FLYVIS_SRC comment in §3.
if getattr(flyvis, "__file__", None) is None:
    raise SystemExit(
        "STOP: `flyvis` resolved to an empty namespace package, not the real one.\n"
        f"Something on sys.path is shadowing it. sys.path[:4] = {sys.path[:4]}\n"
        "Most likely a directory named 'flyvis' next to the notebook's cwd. Restart the\n"
        "runtime and re-run from §1 (§3 now clones to /content/flyvis_repo to avoid this)."
    )

from flyvis import Network

print(f"torch        : {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"flyvis       : {getattr(flyvis, '__version__', '?')}")
print(f"flyvis path  : {Path(flyvis.__file__).parent}")

rc, out = sh(f"git -C {q(FLYVIS_SRC)} rev-parse HEAD", timeout=60)
head = out.strip() if rc == 0 else "(unknown)"
print(f"HEAD         : {head}")
if rc == 0 and head and head != FLYVIS_COMMIT:
    print(f"NOTE: HEAD != the commit this project pinned ({FLYVIS_COMMIT}).")
    print("      Verified API claims were made against that commit; re-check if anything misbehaves.")

net = Network()
con = net.connectome
n_edges = len(con.edges.source_index[:])
n_nodes = len(con.nodes.type[:])
print(f"\nconnectome   : {n_nodes} nodes, {n_edges} edges")

# Attributes our parameterization depends on. Checked, not assumed.
_needed = ("edges_sign", "edges_syn_strength", "nodes_time_const", "nodes_bias", "edges_syn_count")
_missing = [a for a in _needed if not hasattr(net, a)]
for attr in _needed:
    print(f"  Network.{attr:<20} {'ABSENT' if attr in _missing else 'present'}")
if _missing:
    raise SystemExit(f"STOP: flyvis Network lacks {_missing}; the parameterization depends on them.")

save_results(flyvis={"version": getattr(flyvis, "__version__", "?"), "head": head,
                     "n_nodes": int(n_nodes), "n_edges": int(n_edges)})
print("\n[OK] flyvis imports and the connectome loads.")
"""
)

# =============================================================================
md(
    """
## §5 — Reproduce the premise from the released ensemble (free, ~3.4 MB)

The whole project rests on: **task error is stable across retraining seeds, mechanisms are not.**
This reproduces that from the released checkpoints. The archive is small — this is
the cheapest, highest-value cell in the notebook.

Reference values measured independently during review:
`min val loss mean 5.3143, SD 0.0752, CV 1.42%`; per-element parameter CVs `~20% / 98% / 121%`.
"""
)

code(
    r"""
import hashlib, glob, statistics

# The flyvis CLI hardcodes an API key + FOLDER id and *lists* the folder at runtime.
# It does NOT hardcode per-file ids, so we do what it does rather than assume ids.
dl = FLYVIS_SRC / "flyvis_cli" / "download_pretrained_models.py"
print(f"download script present: {dl.exists()}")

pretrained_root = None
try:
    rc, out = sh("flyvis download-pretrained", timeout=1800)
    print(out[-2500:])
    if rc != 0:
        print(f"(exit {rc})")
except Exception as e:
    print("download-pretrained raised:", e)

# Discover what actually landed, rather than assuming a path or a 'best_chkpt' filename.
for base in (Path(flyvis.results_dir) if hasattr(flyvis, "results_dir") else None,
             FLYVIS_SRC / "data" / "results", Path("/content/flyvis_data/results")):
    if base and Path(base).exists():
        pretrained_root = Path(base)
        break

premise = {"ok": False}
if pretrained_root is None:
    print("\nVERDICT (b): could NOT locate downloaded results. §6 and §9 can still run.")
else:
    print(f"\nresults_dir: {pretrained_root}")
    # The glob matches 51 dirs, not 50: the second zip unpacks an `umap_and_clustering`
    # results directory alongside models 000-049. Verified 2026-09-16. Exclude it by name
    # rather than by count, and say so, so the number is 50 for the right reason.
    ens_all = sorted(p for p in pretrained_root.glob("flow/*/*") if p.is_dir())
    non_numeric = [p.name for p in ens_all if not p.name.isdigit()]
    ens = [p for p in ens_all if p.name.isdigit()]
    print(f"models found: {len(ens)}   (glob matched {len(ens_all)})")
    print(f"  names: {[p.name for p in ens[:4]]} ... {[p.name for p in ens[-3:]]}")
    if non_numeric:
        print(f"  NOTE: excluded {len(non_numeric)} non-model dir(s): {non_numeric}")
    chkpt_counts = [len(glob.glob(str(m / 'chkpts' / 'chkpt_*'))) for m in ens[:5]]
    print(f"checkpoints in first 5 models: {chkpt_counts}")

    # Loss parsing. An earlier version wrapped this in `except Exception: pass`, so when
    # it found nothing there was no way to tell whether the files were missing, the key
    # was wrong, or h5py was absent. Never swallow the reason.
    import numpy as np
    losses, parse_errors = [], []
    for m in ens:
        cands = sorted(m.glob("validation/*.h5")) or sorted(m.glob("**/*loss*.h5"))
        if not cands:
            continue
        try:
            import h5py
            with h5py.File(cands[0], "r") as h:
                # flyvis writes validation/loss.h5 as a SCALAR dataset under the key
                # 'data' (shape ()), so `h[k][:]` raises
                #     ValueError: Illegal slicing argument for scalar dataspace
                # `h[k][()]` reads a dataset of ANY shape, and atleast_1d makes the
                # scalar case uniform. Verified on the released ensemble 2026-09-16:
                # model 000 gives [('data', (), 5.136557016107771)].
                vals = []
                for k in h.keys():
                    a = np.atleast_1d(h[k][()])
                    if a.size and np.issubdtype(a.dtype, np.number):
                        vals.append(float(a.min()))
            if vals:
                losses.append(min(vals))
            else:
                parse_errors.append(f"{m.name}: no numeric dataset in {cands[0].name}")
        except Exception as e:
            parse_errors.append(f"{m.name}: {type(e).__name__}: {e}")

    if not losses:
        print("\nDIAGNOSTIC - nothing parsed. Actual contents of the first model dir:")
        if ens:
            for p in sorted(ens[0].rglob("*"))[:30]:
                print("    ", p.relative_to(ens[0]))
        for e in parse_errors[:5]:
            print("     parse error:", e)
    elif parse_errors:
        print(f"\n({len(parse_errors)} model(s) failed to parse; first: {parse_errors[0]})")

    if len(losses) >= 5:
        mean, sd = statistics.fmean(losses), statistics.stdev(losses)
        premise = {"ok": True, "n_models": len(losses), "mean": mean, "sd": sd,
                   "cv_pct": 100 * sd / mean}
        print(f"\nmin val loss over {len(losses)} models: mean {mean:.4f}  SD {sd:.4f}  CV {100*sd/mean:.2f}%")
        print("reference (independent):              mean 5.3143  SD 0.0752  CV 1.42%")
        print("\nVERDICT (b): premise reproduced — task error is stable across seeds.")
    else:
        premise = {"ok": False, "n_models": len(ens), "note": "loss files not parsed"}
        print("\nVERDICT (b): models present but validation losses were not parsed here.")
        print("This is NOT fatal — it does not block §6 or §9.")

save_results(premise=premise, pretrained_root=str(pretrained_root) if pretrained_root else None)
print("\nNOTE: this ensemble is a REFERENCE DISTRIBUTION only. Every released model records")
print("seed: 0 and models 000/007 are byte-identical, so it cannot serve as a baseline arm.")
print("Stage 2 therefore trains TWO arms. See PLAN.md.")
"""
)

# =============================================================================
md(
    """
## §6 — Feasibility: do *our* modules work against the real connectome? (free on CPU)

Two parts, both required for a PASS:

1. **Standalone** (`dale_typed_cfc`): the reference parameterization builds on the real
   connectome, its accounting matches the pre-registration (734 → 929), and each
   treatment is visible *in the tensors*, not only in the arm labels.
2. **The flyvis seam** (`flyvis_seam.verify_seam`): the path Stage 2 actually trains.
   It builds the arms as real `flyvis.Network`s and checks dispatch, parameter counts,
   Dale sign flips, the exact integrator, gate gradients, the inert filler, flyvis's
   optimizer order and a checkpoint round trip.

Everything runs **in this kernel**, after `import flyvis` has set the default device,
so a CUDA-only bug fails here instead of after money is spent. The record is stamped
with the device; §9 only accepts a CUDA pass.

⚠️ If you edit `src/flyvis_seam.py` on Drive mid-session, **Restart session** before
re-running: flyvis dispatches by class name, and a module reload leaves the old classes
live. This cell refuses to continue in that state.
"""
)

code(
    r"""
import importlib, gc

# Clear any earlier verdict FIRST, so a crash below cannot leave a stale ok=True on Drive.
save_results(feasibility={"ok": False, "status": "started", "device": torch.empty(0).device.type})

SRC_CANDIDATES = [WORK / "src", Path("/content/src"), Path.cwd() / "src"]
src_dir = next((p for p in SRC_CANDIDATES if (p / "flyvis_seam.py").exists()), None)
if src_dir is None:
    raise SystemExit(
        "STOP: could not find src/flyvis_seam.py. Upload src/ to the Drive folder "
        f"({WORK}) or to /content/src, then re-run this cell."
    )
print(f"src: {src_dir}")
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
importlib.invalidate_caches()   # Drive files uploaded after the kernel started

_fs_file = src_dir / "flyvis_seam.py"
if "flyvis_seam" in sys.modules and globals().get("_FS_MTIME") != _fs_file.stat().st_mtime:
    raise SystemExit(
        "STOP: src/flyvis_seam.py changed since it was imported in this session. flyvis finds\n"
        "adapter classes by name, and re-importing would leave the old classes live.\n"
        "FIX: Runtime > Restart session (NOT delete runtime), then Run all."
    )
import dale_typed_cfc as dtc
import stability as stab
importlib.reload(dtc); importlib.reload(stab)   # safe: neither registers flyvis classes
import flyvis_seam as fs
_FS_MTIME = _fs_file.stat().st_mtime
DEVICE = torch.empty(0).device.type
print(f"[OK] modules import on the Colab interpreter; default device = {DEVICE}")

# Self-tests in THIS KERNEL'S DEVICE REGIME: each child imports flyvis first, because
# flyvis sets torch's default device at import. A plain `python mod.py` stays on CPU and
# is how a CUDA-only crash (audit K3) once passed its self-test.
for mod in ("dale_typed_cfc", "stability", "flyvis_seam"):
    runner = (
        "import sys, runpy, torch, flyvis; "
        f"sys.path.insert(0, {str(src_dir)!r}); "
        "print('self-test default device:', torch.get_default_device()); "
        f"runpy.run_path({str(src_dir / (mod + '.py'))!r}, run_name='__main__')"
    )
    rc, out = sh(f"{q(sys.executable)} -c {q(runner)}", timeout=900)
    lines = out.strip().splitlines() if out.strip() else ["(no output)"]
    print(f"\n{mod}: exit={rc}")
    # sh() appends stderr (flyvis's import-time logging) after stdout, so the summary is
    # not the last line; show it by content.
    summary = [t for t in lines if "PASSED" in t or "default device" in t]
    for t in (summary or lines[-2:]) if rc == 0 else lines[-15:]:
        print("   ", t)
    if rc != 0:
        raise SystemExit(f"STOP: {mod} self-test failed on Colab. Fix before spending compute.")

# --- part 1: standalone parameterization on the real connectome ---
print("\n--- standalone: building the factorial ladder on the real flyvis connectome ---")
tc = dtc.TypedConnectome.from_flyvis(con)
print(f"TypedConnectome: {tc.n_neurons} neurons, {tc.n_edges} edges, {tc.n_types} types")
mixed = dtc.mixed_sign_types(tc)
print(f"mixed-sign presynaptic types: {sorted(mixed)}")

ladder = dtc.factorial_ladder(tc)
rep = ladder["both"].parameter_report()
base_rep = ladder["baseline"].parameter_report()
filler = ladder["both"].parameter_matched_filler()
print(f"baseline params   : {base_rep['total']}")
print(f"constrained params: {rep['total']}  (+{rep['pct_vs_baseline']:.1f}%)")

one = dtc.assert_shared_substrate(ladder["baseline"], ladder["dale_only"])
both = dtc.assert_shared_substrate(ladder["baseline"], ladder["both"], require_one_variable=False)
print(f"substrate OK; Dale flips {both['n_sign_flips']:,} of {tc.n_edges:,} edge signs (checked on tensors)")

v = torch.zeros(tc.n_neurons)
with torch.no_grad():
    for _ in range(5):
        v = ladder["both"](v)
if not torch.isfinite(v).all():
    raise SystemExit("STOP: standalone 5-step rollout is not finite.")
print(f"5-step standalone rollout on {tc.n_neurons} neurons: finite")

flyvis_trainable = sum(int(p.numel()) for p in net.parameters() if p.requires_grad)
problems = []
def expect(name, got, want):
    if got != want:
        problems.append(f"{name}: got {got!r}, expected {want!r}")
expect("mixed-sign types", sorted(mixed), sorted(dtc.AMBIGUOUS_SOURCE_TYPES))
expect("magnitudes", rep["magnitude"], dtc.FLYVIS.n_type_pairs_checkpoint)
expect("baseline params", base_rep["total"], dtc.FLYVIS.n_learnable)
expect("baseline vs stock flyvis trainable count", base_rep["total"], flyvis_trainable)
expect("constrained params", rep["total"], dtc.ParameterReport.for_fly_connectome().total)
expect("filler", filler, dtc.ParameterReport.for_fly_connectome().delta)
expect("baseline-vs-both axes", sorted(both["differing_axes"]), ["liquid_tau", "sign_rule"])
del v, ladder, tc
gc.collect()

# --- part 2: the flyvis seam, i.e. what Stage 2 trains ---
print("\n--- flyvis seam: verify_seam(extent=15) ---")
seam = fs.verify_seam(verbose=True, extent=15)
sn = seam["numbers"]
if not seam["ok"]:
    problems.append(f"verify_seam failed: {[s['step'] for s in seam['steps'] if not s['ok']]} {seam['error'] or ''}")
else:
    expect("seam trainable (A, dale_only, B)", sn["trainable"], {"A": 929, "dale_only": 929, "B": 929})
    expect("seam Dale flipped groups", sn["dale_plan"]["n_flipped"], 10)

save_results(feasibility={
    "ok": not problems, "problems": problems, "status": "done",
    "device": DEVICE, "gpu": env["gpu_short"],
    "n_neurons": int(n_nodes), "n_edges": int(n_edges), "mixed_sign_types": sorted(mixed),
    "baseline_params": base_rep["total"], "constrained_params": rep["total"],
    "pct_vs_baseline": rep["pct_vs_baseline"], "filler": filler,
    "dale_edge_flips": both["n_sign_flips"],
    "seam_ok": bool(seam["ok"]), "seam_steps": seam["steps"], "seam_numbers": sn,
})
if problems:
    raise SystemExit("STOP: feasibility failed:\n  " + "\n  ".join(problems))
print(f"\n[OK] VERDICT (c): standalone parameterization AND flyvis seam pass on {DEVICE}.")
if DEVICE != "cuda":
    print("NOTE: recorded on CPU. §9 requires this cell to pass again on the GPU runtime.")
"""
)

# =============================================================================
md(
    """
## §7 — Sintel gate (⚠ 5.24 GB download)

Everything above ran without the training dataset. §9 needs it.

**`MPI-Sintel-complete.zip` is 5,627,783,629 bytes over plain HTTP.** flyvis's own
downloader asserts `not sintel_zip.exists()`, so *any* interrupted download permanently
wedges retries — and this account is **Colab Pro, which has no background execution**, so
disconnects mid-download are likely. §8 therefore fetches it itself, resumably, and
repairs a partial file instead of asserting.

⚠️ **Resuming only works on the same VM.** The partial zip lives on `/content`; if the VM
is recycled (idle disconnect, accelerator change) the download restarts from 0. Do
§7–§9 on the GPU runtime, in one session.

Set `I_WANT_SINTEL = True` below to proceed.
"""
)

code(
    r"""
I_WANT_SINTEL = False  # <-- set True to enable §8 and §9

SINTEL_URL = "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"
SINTEL_BYTES = 5_627_783_629

if not I_WANT_SINTEL:
    print("Sintel download SKIPPED. §1-§6 have already answered (a), (b) and (c).")
    print("Set I_WANT_SINTEL = True and re-run this cell to measure cost per run in §9.")
else:
    print(f"Sintel enabled: {SINTEL_BYTES/2**30:.2f} GiB from {SINTEL_URL}")
"""
)

# =============================================================================
md(
    """
## §8 — Sintel pre-flight: resumable, idempotent, bounded (free but slow)

Completion is recorded by a marker file written only after a full, size-verified
extraction. "`training/flow` is non-empty" is NOT evidence of completion: `training/flow`
is the last and largest part of the archive, so an interrupted extraction leaves it
partially filled.
"""
)

code(
    r"""
import urllib.request, urllib.error, zipfile, shutil

sintel_ready = False
if not I_WANT_SINTEL:
    print("skipped (§7)")
else:
    save_results(sintel={"enabled": True, "ready": False})    # an aborted §8 cannot leave ready=True
    sintel_dir = Path(flyvis.sintel_dir)
    sintel_dir.mkdir(parents=True, exist_ok=True)
    zpath = sintel_dir / "MPI-Sintel-complete.zip"
    extracted = sintel_dir / "training" / "flow"
    done_marker = sintel_dir / ".stage0_extract_complete"

    if done_marker.exists():
        print(f"[OK] already extracted at {extracted} (completion marker present)")
        sintel_ready = True
    else:
        have = zpath.stat().st_size if zpath.exists() else 0
        if have > SINTEL_BYTES:
            print(f"oversize zip ({have:,} > {SINTEL_BYTES:,} bytes; a server ignored Range) -> deleting")
            zpath.unlink(); have = 0
        elif have and have != SINTEL_BYTES:
            print(f"partial zip detected: {have:,} / {SINTEL_BYTES:,} bytes -> resuming")
        elif have == SINTEL_BYTES:
            print("[OK] complete zip already on disk")

        MAX_FAILS, deadline, fails = 20, time.time() + 4 * 3600, 0
        while have < SINTEL_BYTES:
            if fails >= MAX_FAILS or time.time() > deadline:
                raise SystemExit(f"STOP: Sintel download gave up at {have:,} bytes after {fails} "
                                 "consecutive failures (or 4 h). Re-run §8 to resume on this VM.")
            start = have
            req = urllib.request.Request(SINTEL_URL, headers={"Range": f"bytes={have}-"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    cr = r.headers.get("Content-Range", "") or ""
                    ranged_ok = (r.status == 206 and cr.startswith(f"bytes {have}-")
                                 and cr.endswith(f"/{SINTEL_BYTES}"))
                    if not (ranged_ok or (have == 0 and r.status == 200)):
                        raise SystemExit(f"STOP: server did not honour Range (HTTP {r.status}, "
                                         f"Content-Range {cr!r}); refusing to append to the partial zip.")
                    with open(zpath, "ab") as f:
                        t0, last = time.time(), have
                        while True:
                            chunk = r.read(1 << 22)
                            if not chunk:
                                break
                            f.write(chunk)
                            have += len(chunk)
                            if time.time() - t0 > 20:
                                mb = (have - last) / 2**20 / (time.time() - t0)
                                print(f"  {have/2**30:6.2f} / {SINTEL_BYTES/2**30:.2f} GiB  ({mb:.1f} MiB/s)")
                                t0, last = time.time(), have
                fails = 0 if have > start else fails + 1
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 403, 404, 410, 416):
                    raise SystemExit(f"STOP: {SINTEL_URL} -> HTTP {e.code}; not retrying.")
                fails += 1
                print(f"  HTTP {e.code} at {have/2**30:.2f} GiB; retry {fails}/{MAX_FAILS}")
                time.sleep(5)
                have = zpath.stat().st_size if zpath.exists() else 0
            except Exception as e:
                fails += 1
                print(f"  interrupted at {have/2**30:.2f} GiB: {e}; retry {fails}/{MAX_FAILS} (resumable)")
                time.sleep(5)
                have = zpath.stat().st_size if zpath.exists() else 0

        if zpath.stat().st_size != SINTEL_BYTES:
            raise SystemExit(f"STOP: zip is {zpath.stat().st_size:,} bytes, expected {SINTEL_BYTES:,}.")
        print("extracting ... (rewrites any partial files from an interrupted run)")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(sintel_dir)
            bad = [i.filename for i in z.infolist()
                   if not i.is_dir() and (not (sintel_dir / i.filename).is_file()
                                          or (sintel_dir / i.filename).stat().st_size != i.file_size)]
        sintel_ready = extracted.exists() and not bad
        if sintel_ready:
            done_marker.write_text(str(SINTEL_BYTES))
        print(f"[OK] extracted -> {extracted}" if sintel_ready
              else f"STOP: extraction incomplete: {len(bad)} members missing/short; re-run §8")

save_results(sintel={"enabled": bool(I_WANT_SINTEL), "ready": bool(sintel_ready)})
"""
)

# =============================================================================
md(
    """
## §9 — Timed training runs, both arms  `# $$$`

**This is the only cell that meaningfully consumes compute units.**

It times **the arms Stage 2 trains** — arm A (exact integrator, softplus magnitudes,
inert filler) and arm B (Dale + CfC) — built through `flyvis_seam`, and refuses to time
anything else. Stock flyvis is a different model and its cost is not our cost.

- The step cap lives in the config (`task.n_iters`). `solver.train(20)` would NOT cap
  it: the first positional argument of `train` is `overfit`.
- Checkpoints (a full validation pass each) are timed separately and added to the
  extrapolation, not mixed into the per-step figure.
- The gates sit *outside* any `try`. They read this VM's disk and a CUDA-stamped §6
  record, never a Drive flag that could be from another VM.
- Live tensors from earlier cells are freed and the remaining allocation is subtracted
  from peak memory.
"""
)

code(
    r"""
import gc, math
from importlib import resources

# ---- GATES (outside any try/except, on purpose) -------------------------
if "fs" not in globals():
    raise SystemExit("GATE: §6 has not run in THIS session (flyvis_seam not imported). Run §6 first.")
_r = load_results()
if not I_WANT_SINTEL:
    raise SystemExit("GATE: §7 not enabled — nothing to time. This is not an error.")
_sd = Path(flyvis.sintel_dir)
_missing = [s for s in ("training", "test", "training/flow", ".stage0_extract_complete") if not (_sd / s).exists()]
if _missing:
    raise SystemExit(f"GATE: Sintel is not on THIS VM's disk ({_sd}; missing {_missing}). Run §8 on "
                     "this runtime first. Refusing to spend compute units.")
_f = _r.get("feasibility", {})
if not _f.get("ok"):
    raise SystemExit("GATE: §6 feasibility did not pass. Refusing to spend compute units.")
if _f.get("device") != "cuda":
    raise SystemExit(f"GATE: §6 passed on device={_f.get('device')!r}, not CUDA. Re-run §6 on this GPU "
                     "runtime first. Refusing to spend compute units.")
if not torch.cuda.is_available():
    raise SystemExit("GATE: no CUDA device — a timing number here would be meaningless.")
if env.get("cu_per_hour", 0.0) <= 0:
    raise SystemExit(f"GATE: no CU/hour rate for GPU {env.get('gpu')!r}; cost per run would read as 0.")
if not fs.register_flyvis_adapters()["registered"]:
    raise SystemExit(f"GATE: flyvis_seam adapters not registered: {fs.SEAM_STATUS['error']}")
print("[OK] all gates passed; proceeding to the metered section.\n")

# Free earlier cells' device tensors so they do not count toward peak memory.
for _name in ("net", "con", "tc", "ladder", "v", "rep", "base_rep", "one", "both", "seam"):
    globals().pop(_name, None)
sys.last_type = sys.last_value = sys.last_traceback = None
gc.collect(); torch.cuda.empty_cache()

from flyvis.solver import MultiTaskSolver
from flyvis.utils.config_utils import get_default_config

SOLVER_YAML = str(resources.files("flyvis") / "config" / "solver.yaml")   # default relative path is broken in 1.2.0
TIMED_STEPS = 20  # small on purpose; we extrapolate
EXPECT = {"A": fs.ExactExponentialDynamics, "B": fs.CfCDynamics}

def arm_config(arm, n_iters):
    base = get_default_config(path=SOLVER_YAML, overrides=[
        "task_name=flow", f"ensemble_and_network_id=stage0_timing/{arm}", f"task.n_iters={n_iters}"])
    return fs.apply_arm_to_solver_config(base, **fs.ARM_PRESETS[arm])

def timed_run(arm, steps):
    # Returns plain numbers only, so no solver stays alive in the caller's scope.
    cfg = arm_config(arm, steps)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    t0 = time.time()
    solver = MultiTaskSolver(config=cfg, delete_if_exists=True)
    setup_s = time.time() - t0
    n = solver.network
    if (type(n.dynamics) is not EXPECT[arm]
            or type(n.edge_params.syn_strength.parameter) is not fs.SoftplusValue
            or n.clamp_config["edges_syn_strength"] is not None
            or n.num_parameters.free != 929):
        raise SystemExit(f"GATE: arm {arm} built {type(n.dynamics).__name__}, "
                         f"{type(n.edge_params.syn_strength.parameter).__name__}, "
                         f"clamp={n.clamp_config['edges_syn_strength']!r}, free={n.num_parameters.free} "
                         "-- NOT the treatment arm; refusing to time it.")
    iters_per_epoch = len(solver.task.train_data)
    chkpt_every_epoch = int(solver.config.scheduler.chkpt_every_epoch)
    real_checkpoint = solver.checkpoint
    solver.checkpoint = lambda: None          # a checkpoint is a full validation pass; timed below
    torch.cuda.synchronize(); t1 = time.time()
    solver.train(overfit=False, initial_checkpoint=False)   # keywords only
    torch.cuda.synchronize(); train_s = time.time() - t1
    n_done = solver.iteration
    if not 0 < n_done < 10 * max(steps, iters_per_epoch):
        raise SystemExit(f"STOP: step cap not applied for arm {arm}: {n_done} iterations.")
    torch.cuda.synchronize(); t2 = time.time()
    real_checkpoint()
    torch.cuda.synchronize(); ckpt_s = time.time() - t2
    peak_gb = (torch.cuda.max_memory_allocated() - base_alloc) / 2**30
    del solver, n
    gc.collect(); torch.cuda.empty_cache()
    return dict(setup_s=setup_s, train_s=train_s, n_done=n_done, per_step_s=train_s / n_done,
                ckpt_s=ckpt_s, peak_gb=peak_gb, iters_per_epoch=iters_per_epoch,
                chkpt_every_epoch=chkpt_every_epoch)

FULL_STEPS = int(get_default_config(path=SOLVER_YAML, overrides=["task_name=flow", "ensemble_and_network_id=0"]).task.n_iters)
cu_hr = env["cu_per_hour"]
per_arm = {}
for arm in ("A", "B"):
    t = timed_run(arm, TIMED_STEPS)
    n_epochs = math.ceil(FULL_STEPS / t["iters_per_epoch"])
    n_ckpts = math.ceil(n_epochs / t["chkpt_every_epoch"]) + 1
    t["full_hours"] = (t["per_step_s"] * FULL_STEPS + t["ckpt_s"] * n_ckpts) / 3600
    t["n_checkpoints_full"] = n_ckpts
    t["cu_per_run"] = t["full_hours"] * cu_hr
    per_arm[arm] = t
    print(f"arm {arm}: setup {t['setup_s']:.1f} s | {t['n_done']} steps {t['train_s']:.1f} s "
          f"({t['per_step_s']:.2f} s/step) | checkpoint {t['ckpt_s']:.1f} s x {n_ckpts} | "
          f"peak {t['peak_gb']:.2f} GiB | full run {t['full_hours']:.1f} h -> {t['cu_per_run']:.1f} CU")

worst = max(per_arm.values(), key=lambda t: t["cu_per_run"])
save_results(timing={
    "per_arm": per_arm, "cu_per_run": worst["cu_per_run"], "full_hours": worst["full_hours"],
    "timed_steps": TIMED_STEPS, "full_steps": FULL_STEPS,
    "gpu": env["gpu"], "gpu_short": env["gpu_short"], "cu_per_hour": cu_hr,
    "flyvis_head": globals().get("head", "?"), "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
})
print(f"\nbudgeting on the costlier arm: {worst['full_hours']:.1f} h -> {worst['cu_per_run']:.1f} CU per run")
print("NOTE: setup (including one-time Sintel rendering on the first arm) is excluded from the per-run")
print("cost. Extrapolating from ~20 steps ignores warm-up and schedule effects: order of magnitude only.")
"""
)

# =============================================================================
md("## §10 — DECISION")

code(
    r"""
r = load_results()
print("=" * 78)
print("STAGE 0 DECISION")
print("=" * 78)

e = r.get("env", {})
print(f"\n(a) GPU              : {e.get('gpu','?')}  @ {e.get('cu_per_hour','?')} CU/h")
p = r.get("premise", {})
print(f"(b) premise          : {'REPRODUCED' if p.get('ok') else 'not reproduced here'}"
      + (f"  (CV {p['cv_pct']:.2f}% over {p['n_models']} models)" if p.get("ok") else ""))
f = r.get("feasibility", {})
print(f"(c) feasibility      : {'PASS' if f.get('ok') else 'NOT ESTABLISHED'}"
      + (f" on {f.get('device')}: standalone {f['baseline_params']} -> {f['constrained_params']} params; "
         f"seam {f['seam_numbers']['trainable']}; Dale flips {f['seam_numbers']['dale_plan']['n_flipped']} groups"
         if f.get("ok") else ""))
print("(e) mechanism dispersion (M2) on real models: NOT MEASURED in Stage 0 -> GO criterion 1 unevaluated")

t = r.get("timing")
if not t:
    print("(d) cost per run     : NOT MEASURED (§9 gated)")
    print("\nNEXT: enable §7 and run §6-§9 on a GPU runtime to get cost per run. Until then N cannot be sized.")
else:
    cu_run = t["cu_per_run"]
    print(f"(d) cost per run     : {t['full_hours']:.1f} h  ->  {cu_run:.1f} CU  (costlier arm; measured on "
          f"{t.get('gpu', 'UNKNOWN')} @ {t.get('cu_per_hour', '?')} CU/h, {t.get('measured_at', '?')})")
    if t.get("gpu_short") != e.get("gpu_short"):
        print(f"NOTE: (d) was measured on {t.get('gpu_short', '?')}, but (a) is this runtime ({e.get('gpu_short', '?')}).")
    USABLE_CU = 242.47      # README: 289.97 minus calibration and a 15% reserve
    FLOOR_RUNS = 2 * 15     # determinism floor, arms A and B (PREREGISTRATION §7)
    ARMS = 2                # two trained arms, unconditionally
    n_per_arm = max(0, int((USABLE_CU - FLOOR_RUNS * cu_run) // (ARMS * max(cu_run, 1e-9))))
    print(f"\n--- BUDGET  ({USABLE_CU} usable of {BUDGET_CU} CU, expiring 90 days after purchase) ---")
    print(f"determinism floor     : {FLOOR_RUNS} runs")
    print(f"arms                  : {ARMS} (unconditional)")
    print(f"seeds per arm         : {n_per_arm}")
    print("NOTE: arm C (Dale on unambiguous types only) is NOT budgeted: on this connectome its signs equal")
    print("      its comparison arm's, so it tests nothing. The pre-registration must be amended first.")
    print("\npower (normal-fingerprint simulation, src/stability.py; heavier-tailed fingerprints need more):")
    for ratio, need in (("4.0x", 24), ("3.0x", 36), ("2.5x", 49), ("2.0x", 80)):
        verdict = "OK" if n_per_arm >= need else "UNDERPOWERED"
        print(f"  variance ratio {ratio}: need {need:>3} per arm  -> {verdict}")
    if n_per_arm < 36:
        print("\nHONEST ASSESSMENT: the budget does not support a well-powered 2-arm design.")
        print("Options, in order of preference:")
        print("  1. Shrink the task (fewer steps/smaller extent) and re-measure cost per run.")
        print("  2. Spread Stage 2 across two months of Pro allocation.")
        print("  3. Ask the Workspace admin about Pro+ (600 CU/mo + background execution).")
        print("  4. Target a 3x effect and accept you cannot detect 2x.")
    else:
        print(f"\nBUDGET OK: {n_per_arm} seeds per arm covers a 3x variance ratio under the normal model.")
        print("Not a full GO: (e) is unmeasured, and PREREGISTRATION P5 requires power on the measured geometry.")
print("\n" + "=" * 78)
print(f"full record: {RESULTS}")
"""
)


# =============================================================================
def build() -> Path:
    nb = {
        "cells": [
            {
                "cell_type": kind,
                "id": f"stage0-{i:02d}",
                "metadata": {},
                "source": body.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for i, (kind, body) in enumerate(CELLS)
        ],
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).parent / "00_stage0_setup_and_benchmark.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    path = build()
    data = json.loads(path.read_text(encoding="utf-8"))
    n_code = sum(1 for c in data["cells"] if c["cell_type"] == "code")
    print(f"wrote {path}")
    print(f"  cells: {len(data['cells'])} ({n_code} code / {len(data['cells']) - n_code} markdown)")
    try:
        import nbformat

        nbformat.validate(nbformat.reads(path.read_text(encoding="utf-8"), as_version=4))
        print("  nbformat.validate: PASS")
    except ImportError:
        print("  nbformat not installed locally; json.load parsed cleanly")
    for c in data["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<cell>", "exec")
    print("  every code cell compiles")
