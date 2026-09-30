"""Does the M2 fingerprint settle before the loss does?  Reads the per-checkpoint M2 files written
by `bench_curve.py long` (m2_long_<arm>.json, one row per checkpoint: iteration, val, fingerprint)
and reports, per arm:

  * validation loss vs iteration (fraction of the total descent reached at each checkpoint)
  * fingerprint distance to the FINAL checkpoint and step-to-step movement
  * distance to the released-model centroid (does training move toward the released cloud?)
  * the first iteration after which the fingerprint stays within r of its final position, for
    r in {0.25, 0.10, 0.05} x (initial distance), against the same for the loss

  python bench/m2_curve_analysis.py runs/long60k/m2_long_A.json runs/long60k/m2_long_B.json
  env M2_ANALYSIS_OUT (json path, optional)
"""
import json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RELEASED = ROOT / "bench" / "results" / "m2_released_local_2026-09-19.json"


def settle_iteration(iters, dist, frac):
    """First iteration after which dist stays <= frac * dist[0] for the rest of the run."""
    ok = dist <= frac * dist[0]
    for k in range(len(ok)):
        if ok[k:].all():
            return int(iters[k])
    return None


def analyze(path):
    d = json.loads(Path(path).read_text())
    rows = d["rows"]
    fps = np.asarray(d["fingerprints"], dtype=np.float64)
    iters = np.asarray([r["iteration"] if r["iteration"] is not None else 0 for r in rows])
    val = np.asarray([r["val"] if r["val"] is not None else np.nan for r in rows], dtype=np.float64)
    order = np.argsort(iters)
    iters, val, fps = iters[order], val[order], fps[order]
    d_final = np.linalg.norm(fps - fps[-1], axis=1)
    step = np.r_[np.nan, np.linalg.norm(np.diff(fps, axis=0), axis=1)]
    out = {"network": d["network"], "n_checkpoints": int(len(rows)), "iterations": iters.tolist(),
           "val": val.tolist(), "dist_to_final": d_final.tolist(), "step": step.tolist()}
    if RELEASED.exists():
        rel = json.loads(RELEASED.read_text())
        assert rel["source_types"] == d["source_types"], "source types differ from the released analysis"
        cen = np.asarray(rel["centroid"])
        d_rel = np.linalg.norm(fps - cen, axis=1)
        out["dist_to_released_centroid"] = d_rel.tolist()
        out["released_mean_dist_to_centroid"] = rel["mean_distance_to_centroid"]
    loss_drop = val[0] - val[-1]
    loss_frac = (val[0] - val) / loss_drop if loss_drop > 0 else np.full_like(val, np.nan)
    loss_remaining = 1 - loss_frac
    out["loss_fraction_of_descent"] = loss_frac.tolist()
    out["settle"] = {}
    for frac in (0.25, 0.10, 0.05):
        out["settle"][str(frac)] = {"fingerprint": settle_iteration(iters, d_final, frac),
                                    "loss": settle_iteration(iters, loss_remaining, frac) if loss_drop > 0 else None}
    return out


def main(paths):
    res = {Path(p).stem: analyze(p) for p in paths}
    for name, a in res.items():
        print(f"\n== {name}  ({a['network']}, {a['n_checkpoints']} checkpoints)")
        print(f"  {'iter':>7s} {'val':>10s} {'loss%':>6s} {'d_final':>8s} {'step':>7s}" +
              ("   d_released" if "dist_to_released_centroid" in a else ""))
        for k in range(a["n_checkpoints"]):
            line = (f"  {a['iterations'][k]:7d} {a['val'][k]:10.3f} {100 * a['loss_fraction_of_descent'][k]:6.1f} "
                    f"{a['dist_to_final'][k]:8.4f} {a['step'][k] if a['step'][k] == a['step'][k] else float('nan'):7.4f}")
            if "dist_to_released_centroid" in a:
                line += f"   {a['dist_to_released_centroid'][k]:.4f}"
            print(line)
        print("  settle (first iteration after which the remaining distance stays below r x initial):")
        for r, s in a["settle"].items():
            print(f"    r={r}: fingerprint {s['fingerprint']}   loss {s['loss']}")
        if "released_mean_dist_to_centroid" in a:
            print(f"  released models' mean distance to their centroid: {a['released_mean_dist_to_centroid']:.4f}")
    out = os.environ.get("M2_ANALYSIS_OUT")
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
