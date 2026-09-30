"""P5: power of the PRIMARY test on the MEASURED fingerprint geometry (PREREGISTRATION section 6.3).

Committed rule: N_dispersion is the smallest N per arm at which simulating this exact pipeline
(fingerprints -> Euclidean distance matrix -> PERMDISP(centroid) -> H1 decision) reaches 80 %
power at a 2x dispersion ratio, calibrated to the measured Stage-0 fingerprint geometry. The
measured geometry is the 50 released-model M2 fingerprints in
bench/results/m2_released_local_2026-09-19.json (24 features, participation ratio 4.85).

Three generators, all calibrated so that the REALIZED total-variance ratio V_A / V_B of the
L1-normalized fingerprints equals the requested ratio (as stability.calibrate_sigma_for_variance_ratio
does for the pre-registered generator):

  prereg    stability's own synthetic generator (PipelineSpec) at the measured dimension and
            effective rank -- the section 6.3 rule exactly as written.
  gaussian  multivariate normal with the measured mean and FULL covariance (anisotropic),
            arm B shrunk toward the mean, re-normalized (L1) as the pipeline would.
  bootstrap the 50 real fingerprints resampled with replacement, arm B shrunk toward the
            real centroid, re-normalized -- the real non-Gaussian cloud, outliers included.

Decision per replicate (PREREGISTRATION section 3.1): PERMDISP p < 0.05 AND arm B's mean
distance to its centroid smaller than arm A's. At ratio 1 the rejection rate is the size.

  python bench/m2_power.py            full grid (n_perm 999) + confirmation of N_dispersion at
                                      ratio 2 with n_perm 9999, n_sims 2000
  python bench/m2_power.py --quick    smaller grid for a smoke run
  env M2_POWER_PROCS (default: cpu_count - 2), M2_POWER_OUT (json path)
"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import stability as st  # noqa: E402

FPS_JSON = ROOT / "bench" / "results" / "m2_released_local_2026-09-19.json"
ALPHA = 0.05
SEED = 20260919
N_GRID = (4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48)
RATIOS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
GENERATORS = ("prereg", "gaussian", "bootstrap")


def l1(x):
    return x / np.abs(x).sum(axis=-1, keepdims=True)


def load_geometry():
    d = json.loads(FPS_JSON.read_text())
    fps = np.asarray(d["fingerprints"], dtype=np.float64)
    return fps, d


# ----------------------------------------------------------------------------------------- generators
class Gen:
    """Draws (n, p) fingerprints for arm A and arm B with realized V_A / V_B = ratio."""

    def __init__(self, kind, fps, ratio, n_ref=4000, seed=SEED):
        self.kind, self.ratio = kind, ratio
        self.fps = fps
        self.mu = fps.mean(axis=0)
        self.cov = np.cov(fps, rowvar=False, ddof=1)
        self.p = fps.shape[1]
        if kind == "prereg":
            pr = float(json.loads(FPS_JSON.read_text())["participation_ratio"])
            self.spec = st.PipelineSpec(n_source_types=12, n_conditions=2, effective_rank=pr)
            cal = st.calibrate_sigma_for_variance_ratio(ratio, spec=self.spec)
            # stability's "arm B" is the MORE dispersed one; here arm B is the constrained (less
            # dispersed) arm, so arm A gets the larger sigma.
            self.sigma_a, self.sigma_b = cal["sigma_b"], cal["sigma_a"]
            self.realized = cal["realized_ratio"]
        else:
            self.shrink = self._calibrate_shrink(ratio, n_ref, seed)

    # --- raw draws of one arm
    def _draw_raw(self, rng, n):
        if self.kind == "gaussian":
            return rng.multivariate_normal(self.mu, self.cov, size=n, method="cholesky")
        idx = rng.integers(0, len(self.fps), n)
        return self.fps[idx]

    def _arm(self, rng, n, shrink):
        raw = self._draw_raw(rng, n)
        x = self.mu + shrink * (raw - self.mu)
        return l1(x)

    def _calibrate_shrink(self, ratio, n_ref, seed):
        """Bisect the shrink factor s so that V(arm B) = V(arm A) / ratio on a large reference
        sample with common random numbers (the pipeline's re-normalization is nonlinear)."""
        rng = np.random.default_rng(seed)
        raw = self._draw_raw(rng, n_ref)

        def V(s):
            return st.ensemble_total_variance(l1(self.mu + s * (raw - self.mu)))

        v_a = V(1.0)
        target = v_a / ratio
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if V(mid) > target:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-5:
                break
        s = 0.5 * (lo + hi)
        self.realized = v_a / V(s)
        self.v_a = v_a
        return s

    def draw(self, rng, n):
        if self.kind == "prereg":
            za = st._pipeline_latents(self.spec, rng, (n,))
            zb = st._pipeline_latents(self.spec, rng, (n,))
            fa = st._fingerprints_from_latents(self.spec, za, self.sigma_a)
            fb = st._fingerprints_from_latents(self.spec, zb, self.sigma_b)
            return fa, fb
        return self._arm(rng, n, 1.0), self._arm(rng, n, self.shrink)


# ------------------------------------------------------------------------------------------ one cell
def run_cell(args):
    kind, ratio, n, n_sims, n_perm, seed = args
    fps, _ = load_geometry()
    gen = Gen(kind, fps, ratio)
    rng = np.random.default_rng([seed, int(ratio * 1000), n, GENERATORS.index(kind), n_perm])
    groups = np.array(["A"] * n + ["B"] * n)
    rej = 0
    disp_ratio = []
    for _ in range(n_sims):
        fa, fb = gen.draw(rng, n)
        D = st.fingerprint_distance_matrix(np.concatenate([fa, fb]), metric="euclidean")
        _, p, disp = st._permdisp_centroid_fast(D, groups, n_perm, rng)
        if p < ALPHA and disp["B"] < disp["A"]:
            rej += 1
        disp_ratio.append(disp["A"] / disp["B"] if disp["B"] > 0 else np.nan)
    return {"generator": kind, "ratio": ratio, "n": n, "n_sims": n_sims, "n_perm": n_perm,
            "power": rej / n_sims, "realized_ratio": float(gen.realized),
            "mean_disp_ratio": float(np.nanmean(disp_ratio))}


def n_for_power(cells, target=0.8):
    """Smallest n on the grid whose power >= target (None if the grid does not reach it)."""
    for c in sorted(cells, key=lambda c: c["n"]):
        if c["power"] >= target:
            return c["n"], c["power"]
    return None, max(c["power"] for c in cells)


def main(quick=False):
    fps, meta = load_geometry()
    t0 = time.time()
    n_grid = (6, 12, 24, 36) if quick else N_GRID
    ratios = (1.0, 2.0, 3.0) if quick else RATIOS
    n_sims, n_perm = (200, 499) if quick else (500, 999)
    procs = int(os.environ.get("M2_POWER_PROCS", max(1, (os.cpu_count() or 4) - 2)))
    cells = [(g, r, n, n_sims, n_perm, SEED) for g in GENERATORS for r in ratios for n in n_grid]
    print(f"[power] geometry: {fps.shape[0]} models x {fps.shape[1]} features, V={meta['total_variance']:.4f}, "
          f"PR={meta['participation_ratio']:.2f}; {len(cells)} cells x {n_sims} sims, {procs} procs", flush=True)
    with Pool(procs) as pool:
        results = pool.map(run_cell, cells, chunksize=1)
    print(f"[power] grid done in {time.time() - t0:.0f} s", flush=True)

    # confirmation at the pre-registered n_perm for the ratio-2 cells around N_dispersion
    confirm = []
    if not quick:
        conf_cells = []
        for g in GENERATORS:
            sub = [c for c in results if c["generator"] == g and c["ratio"] == 2.0]
            n80, _ = n_for_power(sub)
            if n80 is not None:
                i = N_GRID.index(n80)
                for n in sorted({N_GRID[max(i - 1, 0)], n80, N_GRID[min(i + 1, len(N_GRID) - 1)]}):
                    conf_cells.append((g, 2.0, n, 2000, 9999, SEED + 1))
        with Pool(procs) as pool:
            confirm = pool.map(run_cell, conf_cells, chunksize=1)
        print(f"[power] confirmation done in {time.time() - t0:.0f} s", flush=True)

    # summary
    summary = {}
    for g in GENERATORS:
        sg = {}
        for r in ratios:
            sub = [c for c in results if c["generator"] == g and c["ratio"] == r]
            n80, best = n_for_power(sub, 0.8)
            n90, _ = n_for_power(sub, 0.9)
            sg[str(r)] = {"n_80": n80, "n_90": n90, "max_power_on_grid": best,
                          "realized_ratio": sub[0]["realized_ratio"],
                          "power_by_n": {str(c["n"]): c["power"] for c in sorted(sub, key=lambda c: c["n"])}}
        conf = sorted([c for c in confirm if c["generator"] == g], key=lambda c: c["n"])
        n_disp = next((c["n"] for c in conf if c["power"] >= 0.8), None) if conf else sg["2.0"]["n_80"]
        summary[g] = {"by_ratio": sg, "confirmation_ratio2": [{k: c[k] for k in ("n", "power", "n_sims", "n_perm")} for c in conf],
                      "N_dispersion_ratio2": n_disp,
                      "size_at_ratio1": sg["1.0"]["power_by_n"]}
    out = {
        "what": "PREREGISTRATION 6.3 / P5: PERMDISP power on the measured M2 geometry (50 released models)",
        "decision": "PERMDISP(centroid) p < 0.05 and arm B less dispersed",
        "alpha": ALPHA, "geometry_source": str(FPS_JSON.name),
        "geometry": {"n_models": int(fps.shape[0]), "n_features": int(fps.shape[1]),
                     "total_variance": meta["total_variance"], "participation_ratio": meta["participation_ratio"],
                     "centroid_norm": meta["centroid_norm"]},
        "grid": {"n": list(n_grid), "ratios": list(ratios), "n_sims": n_sims, "n_perm": n_perm},
        "cells": results, "confirmation": confirm, "summary": summary,
        "seconds": time.time() - t0,
    }
    path = Path(os.environ.get("M2_POWER_OUT", ROOT / "bench" / "results" / "m2_power_2026-09-19.json"))
    path.write_text(json.dumps(out, indent=1))

    print("\nPOWER (rows: N per arm; columns: V_A/V_B ratio; decision = p<0.05 and B less dispersed)")
    for g in GENERATORS:
        print(f"\n  generator = {g}   realized ratios: " +
              ", ".join(f"{r:g}->{summary[g]['by_ratio'][str(r)]['realized_ratio']:.2f}" for r in ratios))
        print("   N   " + "".join(f"{r:>7g}" for r in ratios))
        for n in n_grid:
            row = [summary[g]["by_ratio"][str(r)]["power_by_n"][str(n)] for r in ratios]
            print(f"  {n:3d}  " + "".join(f"{p:7.2f}" for p in row))
        print("   N_80: " + ", ".join(f"{r:g}x:{summary[g]['by_ratio'][str(r)]['n_80']}" for r in ratios))
        if summary[g]["confirmation_ratio2"]:
            print("   confirmation at ratio 2 (n_perm 9999, 2000 sims): " +
                  ", ".join(f"n={c['n']}:{c['power']:.3f}" for c in summary[g]["confirmation_ratio2"]))
        print(f"   => N_dispersion (ratio 2, 80 %) = {summary[g]['N_dispersion_ratio2']}")
    print(f"\n[power] wrote {path}  ({out['seconds']:.0f} s)")
    print("POWER DONE")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
