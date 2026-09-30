"""M2 mechanism fingerprint on REAL trained models: the released flyvis ensemble.

Pre-registered fingerprint (PREREGISTRATION.md section 3, stability.PRIMARY_ANALYSIS): signed,
time-integrated currents from every presynaptic type into the central T4c cell, for a moving
edge at intensity 1 in the preferred (90 deg) and null (270 deg) direction, window [0, 1] s after
stimulus onset, dt 1/200, jointly L1-normalized over (source type x direction). All source types
with an edge into T4c, in flyvis's own order, hide_source_types=None.

This answers Stage 0 question (e) / kill criterion 1: is the across-seed dispersion of the
MECHANISM large relative to the dispersion of TASK ERROR (CV 1.42 % over the 50 released models)?
It also freezes the source-type list (P2) and yields the real fingerprint geometry for the power
calculation (P5).

  python m2_released.py compute OUT.json [ids...]    lean path (default: all 50 released models)
  python m2_released.py reference OUT.json ids...    flyvis's own moving_edge_currents path, for
                                                     cross-checking `compute` (12 stimuli, slow)
  python m2_released.py analyze OUT.json IN.json...  merge shards -> mechanism CV, GO criterion 1,
                                                     fingerprint geometry
  env: FLYVIS_ROOT_DIR (results/flow/0000 must exist or be downloadable), M2_THREADS (torch
       intra-op threads per process, for running shards side by side), M2_PREFIX (default flow/0000)

The lean path renders only the two (angle, intensity) sequences the fingerprint reads, once, and
rebuilds the currents on the central T4c cell's input edges from the activity trace with the
dynamics' own formula, current_e = weight_e * activation(activity[source_e])
(flyvis PPNeuronIGRSynapses.currents, dynamics.py:235; weight = sign*syn_count*syn_strength,
:166). flyvis's path (`reference`) materialises all 1.5 M edge currents per frame (2.9 GB per
stimulus) and re-instantiates the 12-stimulus dataset per model; the numbers are the same, which
`reference` on a couple of models verifies.
"""
import json, os, sys, time, subprocess
from pathlib import Path

os.environ.setdefault("FLYVIS_ROOT_DIR", "/content/flyvis_data")
SRC = Path(__file__).resolve().parent.parent / "src"
for p in (str(SRC), "/content", str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import flyvis
from flyvis import NetworkView
from flyvis.analysis.stimulus_responses_currents import ExperimentData, TargetData, moving_edge_currents
from flyvis.analysis.moving_edge_currents import MovingEdgeCurrentView, reset_index
from flyvis.connectome.connectome import ReceptiveFields
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.utils.activity_utils import LayerActivity
import stability as st

TARGET = "T4c"
ANGLE_PD, CONTRAST, DT, T0, T1 = 90.0, 1.0, 1 / 200, 0.0, 1.0
ANGLE_ND = (ANGLE_PD - 180) % 360
T_PRE_SIM = 1.0                        # flyvis moving_edge_currents: 1 s of grey 0.5 before the stimulus
TASK_CV_RELEASED = 0.0752 / 5.3143     # SD/mean of stored min validation loss, 50 models (ddof=1)
PREFIX = os.environ.get("M2_PREFIX", "flow/0000")

# flyvis moving_edge_currents' default_dataset_config (stimulus_responses_currents.py:205-218),
# restricted to the two (angle, intensity) rows the fingerprint reads. MovingEdge renders every
# (angle, intensity) sequence independently (RenderedOffsets) and with a single speed there is no
# cross-sequence padding, so these two sequences are identical to rows 5 and 11 of the 12-stimulus set.
STIMULUS_FULL = dict(widths=[80], offsets=(-10, 11), intensities=[0, 1], speeds=(19,), height=80,
                     bar_loc_horizontal=0.0, shuffle_offsets=False, post_pad_mode="continue",
                     t_pre=1.0, t_post=1.0, dt=DT, angles=(0, 45, 90, 180, 225, 270))
STIMULUS_LEAN = dict(STIMULUS_FULL, intensities=[CONTRAST], angles=[ANGLE_PD, ANGLE_ND])
SETTINGS = {"target": TARGET, "angle_pd": ANGLE_PD, "angle_nd": ANGLE_ND, "contrast": CONTRAST, "dt": DT,
            "t_start": T0, "t_end": T1, "t_pre_sim": T_PRE_SIM, "reduce": "integral", "normalize": "l1-joint",
            "hide_source_types": None, "stimulus": STIMULUS_FULL, "flyvis": flyvis.__version__}


def log(*a):
    print(*a, flush=True)


def patch_datamate_for_windows():
    """datamate 1.0.0 io._write_h5 opens the target with mode='w' inside its try block and, when the
    fresh file has no 'data' dataset, unlinks it while that handle is still open -> WinError 32 on
    Windows (Linux unlinks open files fine). Close the handle first."""
    if os.name != "nt":
        return
    import h5py as h5
    import datamate.io, datamate.directory

    def _write_h5(path, val):
        val = np.asarray(val)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            path.rmdir()
        elif path.exists():
            path.unlink()
        with h5.File(path, libver="latest", mode="w") as f:
            f["data"] = val
            f.swmr_mode = True

    datamate.io._write_h5 = _write_h5
    datamate.directory._write_h5 = _write_h5


def ensure_pretrained():
    if not (Path(flyvis.results_dir) / PREFIX / "000").exists():
        log("[m2] downloading pretrained models ...")
        subprocess.run(["flyvis", "download-pretrained"], check=True)


def load_views(models):
    kw = dict(best_checkpoint_fn_kwargs={"validation_subdir": "validation", "loss_file_name": "loss"})
    return {mid: NetworkView(f"{PREFIX}/{mid}", **kw) for mid in models}


def frozen_source_types(view: NetworkView):
    """All presynaptic types with >= 1 edge into T4c, in flyvis's own ReceptiveFields order.
    Returns (rfs with global edge ids, rfs with reset index as MovingEdgeCurrentView uses, names)."""
    rfs_raw = ReceptiveFields(TARGET, view.connectome.edges.to_df())
    rfs = reset_index(rfs_raw)
    return rfs_raw, rfs, tuple(rfs.source_types)


def contributions_from(exp_data, rfs, source_types, arg_df=None):
    """(n_models, n_types, 2) signed time integrals [A*s] via the pre-registered stability path."""
    cv = MovingEdgeCurrentView(None, TARGET, exp_data, arg_df=arg_df, rfs=rfs)
    ens = st.CurrentEnsemble.from_flyvis_current_view(
        cv, source_types, contrast=CONTRAST, angle=ANGLE_PD, dt=DT, t_start=T0, t_end=T1,
        reduce="integral", arm="released")
    C = np.asarray(ens.contributions, dtype=np.float64)
    assert np.isfinite(C).all()
    return C


def dump(path, models, source_types, C, seconds, mode, checkpoints):
    out = {"what": f"M2 contributions of released flyvis models ({PREFIX}), 1.2.0 code, mode={mode}",
           "settings": SETTINGS, "mode": mode, "models": list(models), "checkpoints": checkpoints,
           "source_types": list(source_types), "contributions": C.tolist(),
           "seconds_per_model": seconds, "device": str(flyvis.device)}
    Path(path).write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------------------------- compute
def central_input_currents(net, dataset, rfs_raw):
    """Lean equivalent of flyvis compute_currents for the central TARGET cell: simulate once (activity
    only) and rebuild the currents on the target's input edges per source type."""
    with torch.no_grad():
        net.clamp()                                   # current_response clamps before building params
        params = net._param_api()
        w = params.edges.weight
        weight = (w.deref() if hasattr(w, "deref") else w).detach().cpu()          # (n_edges,) float32
    # explicit CPU: flyvis sets torch's default device to CUDA at import, and the activity trace
    # below is a CPU array (indexing a CPU tensor with CUDA indices raises)
    src_index = torch.as_tensor(np.asarray(net.connectome.edges.source_index[:]), dtype=torch.long, device="cpu")
    edge_ids = {s: torch.as_tensor(np.asarray(rfs_raw[s].index), dtype=torch.long, device="cpu")
                for s in rfs_raw.source_types}
    act_fn = net.dynamics.activation
    indexer = LayerActivity(None, net.connectome, keepref=True)
    td = TargetData()
    gen = net.stimulus_response(dataset, DT, indices=None, t_pre=T_PRE_SIM, t_fade_in=0.0, batch_size=1)
    for _stim, activity in gen:                                                     # (1, n_frames, n_nodes)
        activity = np.ascontiguousarray(activity[0])
        indexer.update(activity)
        td.activity_central.append(np.asarray(indexer.central[TARGET]))
        a = torch.from_numpy(activity)
        with torch.no_grad():
            for s in rfs_raw.source_types:
                e = edge_ids[s]
                cur = (weight[e] * act_fn(a[:, src_index[e]])).numpy()              # == current[..., e]
                td.source_data.setdefault(s, []).append(cur)
    return td


def compute(out_path, models):
    ensure_pretrained()
    patch_datamate_for_windows()
    if os.environ.get("M2_THREADS"):
        torch.set_num_threads(int(os.environ["M2_THREADS"]))
    views = load_views(models)
    first = next(iter(views.values()))
    rfs_raw, rfs, source_types = frozen_source_types(first)
    log(f"[m2] T4c source types (flyvis order, n={len(source_types)}): {list(source_types)}")
    t0 = time.time()
    dataset = MovingEdge(**STIMULUS_LEAN)
    log(f"[m2] stimulus ready in {time.time() - t0:.0f} s: {len(dataset)} sequences, "
        f"{dataset[0].shape[0]} frames, arg_df=\n{dataset.arg_df}")
    net = None
    exp_data, done, seconds, ckpts = [], [], [], []
    for mid, nv in views.items():
        t0 = time.time()
        ckpt = nv.get_checkpoint("best")
        if net is None:
            net = nv.network_class(**nv.dir.config.network.to_dict())
        nv.recover_fn(net, ckpt)
        net.eval()
        td = central_input_currents(net, dataset, rfs_raw)
        exp_data.append(ExperimentData(config=dataset.config, checkpoint=str(ckpt), target_data={TARGET: td}))
        done.append(mid); seconds.append(time.time() - t0); ckpts.append(str(ckpt))
        C = contributions_from(exp_data, rfs, source_types, arg_df=dataset.arg_df)
        dump(out_path, done, source_types, C, seconds, "lean", ckpts)
        fp = st.mechanism_fingerprint(C)[-1]
        top = sorted(zip(source_types, C[-1, :, 0], C[-1, :, 1]), key=lambda t: -abs(t[1]))[:3]
        log(f"[m2] {mid} done in {seconds[-1]:.0f} s | top PD: " +
            ", ".join(f"{s} {pd:+.3e}/{nd:+.3e}" for s, pd, nd in top) + f" | fp L1={np.abs(fp).sum():.3f}")
    log(f"[m2] wrote {out_path} ({len(done)} models, {np.mean(seconds):.0f} s/model)")
    log("M2 COMPUTE DONE")


# --------------------------------------------------------------------------------------------- reference
def reference(out_path, models):
    """flyvis's own path: NetworkView.moving_edge_currents -> compute_currents -> current_response."""
    ensure_pretrained()
    patch_datamate_for_windows()
    views = load_views(models)
    first = next(iter(views.values()))
    _, rfs, source_types = frozen_source_types(first)
    t0 = time.time()
    exp_data = moving_edge_currents(views, target_cell_types=[TARGET], dt=DT)      # 12 stimuli per model
    t_cur = time.time() - t0
    log(f"[m2] reference currents for {len(exp_data)} models in {t_cur:.0f} s ({t_cur / len(exp_data):.0f} s/model)")
    C = contributions_from(exp_data, rfs, source_types)
    dump(out_path, list(models), source_types, C, [t_cur / len(exp_data)] * len(exp_data), "flyvis",
         [str(e.checkpoint) for e in exp_data])
    log(f"[m2] wrote {out_path}")
    log("M2 REFERENCE DONE")


# ------------------------------------------------------------------------------------------- checkpoints
def checkpoints(out_path, network_name):
    """M2 on EVERY checkpoint of one trained network dir (ours: results/flow/curve/<tag>), for the
    question 'does the fingerprint settle before the loss does'. Seam arms need the adapters."""
    patch_datamate_for_windows()
    try:
        import flyvis_seam as fs
        log(f"[m2] seam adapters: {fs.register_flyvis_adapters()}")
    except Exception as e:
        log(f"[m2] seam adapters not available ({e}); stock flyvis classes only")
    nv = NetworkView(network_name, best_checkpoint_fn_kwargs={"validation_subdir": "validation", "loss_file_name": "loss"})
    rfs_raw, rfs, source_types = frozen_source_types(nv)
    dataset = MovingEdge(**STIMULUS_LEAN)
    net = nv.network_class(**nv.dir.config.network.to_dict())
    def record(node):
        try:
            return [float(v) for v in np.asarray(node[:]).ravel()]
        except Exception:
            return []           # released dirs ship no chkpt_iter
    iters = [int(i) for i in record(nv.dir.chkpt_iter)]
    vals = record(nv.dir.validation.loss)
    ck = nv.checkpoints
    pairs = list(enumerate(zip(ck.indices, ck.paths)))          # (global position, (chkpt id, path))
    sl = os.environ.get("M2_CKPT_SLICE")                          # "start:stop" over positions, for shards
    if sl:
        a, b = sl.split(":")
        pairs = pairs[int(a or 0):(int(b) if b else None)]
        log(f"[m2] shard positions {sl}: {len(pairs)} of {len(ck.indices)} checkpoints")
    exp_data, rows = [], []
    t0 = time.time()
    for k, (idx, path) in pairs:
        nv.recover_fn(net, path)
        net.eval()
        td = central_input_currents(net, dataset, rfs_raw)
        exp_data.append(ExperimentData(config=dataset.config, checkpoint=str(path), target_data={TARGET: td}))
        rows.append({"chkpt": int(idx), "iteration": max(iters[k], 0) if k < len(iters) else None,
                     "val": vals[k] if k < len(vals) else None})
        log(f"[m2] {network_name} chkpt {idx} iter {rows[-1]['iteration']} ({time.time() - t0:.0f} s)")
    C = contributions_from(exp_data, rfs, source_types, arg_df=dataset.arg_df)
    write_checkpoint_file(out_path, network_name, source_types, rows, C)
    log("M2 CHECKPOINTS DONE")


def write_checkpoint_file(out_path, network_name, source_types, rows, C):
    fps = st.mechanism_fingerprint(C)
    d_final = np.linalg.norm(fps - fps[-1], axis=1)
    d_step = np.r_[np.nan, np.linalg.norm(np.diff(fps, axis=0), axis=1)]
    mass = np.abs(C).sum(axis=(1, 2))
    for r, a, b, m in zip(rows, d_final, d_step, mass):
        r.update({"dist_to_final": float(a), "step": float(b) if np.isfinite(b) else None, "l1_mass": float(m)})
    out = {"what": f"M2 fingerprint at every checkpoint of {network_name} (pre-registered settings)",
           "network": network_name, "settings": SETTINGS, "source_types": list(source_types),
           "rows": rows, "contributions": C.tolist(), "fingerprints": fps.tolist(), "device": str(flyvis.device)}
    Path(out_path).write_text(json.dumps(out, indent=1))
    for r in rows:
        log(f"M2CURVE net={network_name} chkpt={r['chkpt']} iter={r['iteration']} val={r['val']} "
            f"dist_to_final={r['dist_to_final']:.4f} step={r['step']} mass={r['l1_mass']:.3f}")
    log(f"[m2] wrote {out_path}")


def merge_checkpoint_shards(out_path, in_paths):
    """Concatenate `checkpoints` shard files of ONE network (sorted by chkpt id) into one file;
    distance-to-final and step are recomputed on the merged sequence."""
    rows, C, net, sts = [], [], None, None
    for p in in_paths:
        d = json.loads(Path(p).read_text())
        if net is None:
            net, sts = d["network"], tuple(d["source_types"])
        assert d["network"] == net and tuple(d["source_types"]) == sts, f"{p}: different network or source types"
        rows += d["rows"]; C += d["contributions"]
    order = np.argsort([r["chkpt"] for r in rows])
    rows = [{k: rows[i][k] for k in ("chkpt", "iteration", "val")} for i in order]
    write_checkpoint_file(out_path, net, sts, rows, np.asarray(C, dtype=np.float64)[order])
    log("M2 MERGE DONE")


# ----------------------------------------------------------------------------------------------- analyze
def analyze(out_path, in_paths):
    rows, source_types, settings, seconds = {}, None, None, []
    for p in in_paths:
        d = json.loads(Path(p).read_text())
        if source_types is None:
            source_types, settings = tuple(d["source_types"]), d["settings"]
        assert tuple(d["source_types"]) == source_types, f"{p}: source types differ"
        for mid, c in zip(d["models"], d["contributions"]):
            rows[mid] = c
        seconds += list(d.get("seconds_per_model", []))
    models = sorted(rows)
    C = np.asarray([rows[m] for m in models], dtype=np.float64)   # (n_models, n_types, 2) signed, A*s
    n, n_types = C.shape[0], C.shape[1]
    fps = st.mechanism_fingerprint(C)                              # joint L1, layout [s0_PD, s0_ND, s1_PD, ...]
    assert np.isfinite(fps).all()

    V = st.ensemble_total_variance(fps)
    centroid = fps.mean(axis=0)
    z = np.linalg.norm(fps - centroid, axis=1)
    cnorm = float(np.linalg.norm(centroid))
    mech_cv = float(np.sqrt(V) / cnorm)
    D = st.fingerprint_distance_matrix(fps, metric="euclidean")
    rng = np.random.default_rng(20260915)
    boot = []
    for _ in range(2000 if n >= 5 else 0):
        f = fps[rng.integers(0, n, n)]
        boot.append(np.sqrt(st.ensemble_total_variance(f)) / np.linalg.norm(f.mean(axis=0)))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (None, None)

    # geometry for the power calculation: covariance spectrum, per-feature variance share
    X = fps - centroid
    cov = X.T @ X / max(n - 1, 1)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eig = np.clip(eig, 0, None)
    pr = float(eig.sum() ** 2 / (eig ** 2).sum()) if eig.sum() > 0 else float("nan")
    var_share = (X ** 2).sum(axis=0) / max((X ** 2).sum(), 1e-300)
    feat = [f"{s}_{c}" for s in source_types for c in ("PD", "ND")]

    per_type = {}
    for i, s in enumerate(source_types):
        d = {"PD_mean": float(C[:, i, 0].mean()), "PD_sd": float(C[:, i, 0].std(ddof=1)) if n > 1 else 0.0,
             "ND_mean": float(C[:, i, 1].mean()), "ND_sd": float(C[:, i, 1].std(ddof=1)) if n > 1 else 0.0,
             "fp_PD_mean": float(fps[:, 2 * i].mean()), "fp_PD_sd": float(fps[:, 2 * i].std(ddof=1)) if n > 1 else 0.0,
             "fp_ND_mean": float(fps[:, 2 * i + 1].mean()), "fp_ND_sd": float(fps[:, 2 * i + 1].std(ddof=1)) if n > 1 else 0.0,
             "var_share_PD": float(var_share[2 * i]), "var_share_ND": float(var_share[2 * i + 1])}
        d["PD_cv"] = abs(d["PD_sd"] / d["PD_mean"]) if d["PD_mean"] else None
        d["ND_cv"] = abs(d["ND_sd"] / d["ND_mean"]) if d["ND_mean"] else None
        per_type[s] = d

    out = {
        "what": f"M2 fingerprint of released flyvis models ({PREFIX}), 1.2.0 code, pre-registered settings",
        "settings": settings, "inputs": [str(p) for p in in_paths],
        "models": models, "source_types": list(source_types), "features": feat,
        "contributions": C.tolist(), "fingerprints": fps.tolist(),
        "distance_to_centroid": z.tolist(),
        "total_variance": float(V), "mean_distance_to_centroid": float(z.mean()),
        "distance_to_centroid_quantiles": {q: float(np.percentile(z, q)) for q in (0, 25, 50, 75, 100)},
        "centroid_norm": cnorm, "centroid": centroid.tolist(),
        "mechanism_cv": mech_cv, "mechanism_cv_ci95_bootstrap": ci,
        "task_cv_released": TASK_CV_RELEASED, "ratio_mech_to_task_cv": mech_cv / TASK_CV_RELEASED,
        "go_criterion_1_ratio_ge_5": bool(mech_cv >= 5 * TASK_CV_RELEASED),
        "mean_pairwise_distance": float(D[np.triu_indices(n, 1)].mean()) if n > 1 else None,
        "max_pairwise_distance": float(D.max()) if n > 1 else None,
        "cov_eigenvalues": eig.tolist(), "participation_ratio": pr,
        "top_variance_features": [(feat[i], float(var_share[i])) for i in np.argsort(-var_share)[:6]],
        "per_source_type": per_type,
        "seconds_per_model": float(np.mean(seconds)) if seconds else None,
    }
    Path(out_path).write_text(json.dumps(out, indent=1))
    log(f"RESULT m2 n_models={n} n_types={n_types} mech_cv={mech_cv:.4f} ci95=({ci[0]:.4f}, {ci[1]:.4f}) "
        f"task_cv={TASK_CV_RELEASED:.4f} ratio={mech_cv / TASK_CV_RELEASED:.2f}x GO1={out['go_criterion_1_ratio_ge_5']} "
        f"mean_z={z.mean():.4f} V={V:.3e} centroid_norm={cnorm:.4f} PR={pr:.2f} mean_pair={out['mean_pairwise_distance']:.4f}")
    log(f"  {'type':10s} {'PD [A*s]':>22s} {'ND [A*s]':>22s}   {'fp PD':>15s} {'fp ND':>15s}  var-share")
    for s in source_types:
        p = per_type[s]
        log(f"  {s:10s} {p['PD_mean']:+.3e} ± {p['PD_sd']:.1e} {p['ND_mean']:+.3e} ± {p['ND_sd']:.1e}   "
            f"{p['fp_PD_mean']:+.3f}±{p['fp_PD_sd']:.3f} {p['fp_ND_mean']:+.3f}±{p['fp_ND_sd']:.3f}  "
            f"{100 * p['var_share_PD']:4.1f}% {100 * p['var_share_ND']:4.1f}%")
    log(f"[m2] wrote {out_path}")
    log("M2 ANALYZE DONE")


if __name__ == "__main__":
    cmd, rest = (sys.argv[1] if len(sys.argv) > 1 else "compute"), sys.argv[2:]
    if cmd == "compute":
        compute(rest[0] if rest else "m2_released_contrib.json", rest[1:] or [f"{i:03d}" for i in range(50)])
    elif cmd == "reference":
        reference(rest[0], rest[1:])
    elif cmd == "analyze":
        analyze(rest[0], rest[1:])
    elif cmd == "checkpoints":
        checkpoints(rest[0], rest[1])
    elif cmd == "merge":
        merge_checkpoint_shards(rest[0], rest[1:])
    else:
        sys.exit(__doc__)
