"""Direction tuning of the central T4 cells of one released flyvis model, for the explainer.

Moving bright edge (flyvis MovingEdge, width 80, speed 19, intensity 1) at 12 angles 0..330;
records the central T4a/b/c/d (and T5a-d) voltage over time and summarises each angle by the peak
and by the time integral of relu(v) over [0, 1] s after onset.

  python tuning_released.py OUT.json [model_id]     (default model 000)
  env FLYVIS_ROOT_DIR must point at a root with results/flow/0000/<id>
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "src")]
import numpy as np
import torch
import flyvis
from flyvis import NetworkView
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.utils.activity_utils import LayerActivity
import m2_released as m2

TYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
ANGLES = list(range(0, 360, 30))
DT, T_PRE = 1 / 200, 1.0


def main(out_path, mid="000"):
    m2.patch_datamate_for_windows()
    nv = NetworkView(f"flow/0000/{mid}", best_checkpoint_fn_kwargs={"validation_subdir": "validation", "loss_file_name": "loss"})
    net = nv.init_network()
    net.eval()
    t0 = time.time()
    ds = MovingEdge(**dict(m2.STIMULUS_FULL, intensities=[1], angles=ANGLES))
    print(f"[tuning] stimulus: {len(ds)} sequences, {ds[0].shape[0]} frames ({time.time() - t0:.0f} s)", flush=True)
    idx = LayerActivity(None, net.connectome, keepref=True)
    traces = {t: [] for t in TYPES}
    angles_seen = []
    for k, (_stim, act) in enumerate(net.stimulus_response(ds, DT, indices=None, t_pre=T_PRE, t_fade_in=0.0, batch_size=1)):
        a = np.ascontiguousarray(act[0])
        idx.update(a)
        angles_seen.append(float(ds.arg_df.angle.iloc[k]))
        for t in TYPES:
            traces[t].append(np.asarray(idx.central[t], dtype=np.float64))
        print(f"[tuning] angle {angles_seen[-1]:.0f} done ({time.time() - t0:.0f} s)", flush=True)
    n_frames = len(traces[TYPES[0]][0])
    time_s = np.arange(n_frames) * DT - T_PRE
    win = (time_s >= 0) & (time_s < 1.0)
    out = {"model": mid, "angles": angles_seen, "dt": DT, "t_pre": T_PRE, "n_frames": int(n_frames),
           "time_s_subsampled": time_s[::5].round(4).tolist(), "types": TYPES, "traces": {}, "peak": {}, "integral": {}}
    for t in TYPES:
        arr = np.stack(traces[t])                                  # (n_angles, n_frames)
        out["traces"][t] = arr[:, ::5].round(5).tolist()
        out["peak"][t] = np.clip(arr[:, win], 0, None).max(axis=1).round(5).tolist()
        out["integral"][t] = (np.clip(arr[:, win], 0, None).sum(axis=1) * DT).round(5).tolist()
    Path(out_path).write_text(json.dumps(out))
    for t in TYPES:
        pk = np.array(out["peak"][t]); best = int(np.argmax(pk))
        print(f"[tuning] {t}: preferred angle {angles_seen[best]:.0f} (peak {pk[best]:.4f}), "
              f"opposite {pk[(best + 6) % 12]:.4f}, DSI {(pk[best] - pk[(best + 6) % 12]) / (pk[best] + pk[(best + 6) % 12] + 1e-12):.2f}", flush=True)
    print(f"[tuning] wrote {out_path} ({time.time() - t0:.0f} s)")
    print("TUNING DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "000")
