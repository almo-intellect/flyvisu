# Findings log

Running record of what has been measured, fixed and decided. Newest at the bottom. Numbers are
quoted as measured; where they are estimates, it says so. Companion documents: `PLAN.md`
(reasoning and budget), `PREREGISTRATION.md` (the commitment), `CRITIQUE_ROUND1.md` (round-1
review). Visual summaries: project map https://claude.ai/artifact/J4M5Nk6UYNnKH1wQRWgApv,
architecture explainer https://claude.ai/artifact/X449UaExdYUmhQihd45i6r.

## Environment facts (Colab Pro, Workspace-managed account)

| Fact | Value | Source |
|---|---|---|
| Runtime version flyvis needs | `2026.07` (Python 3.12.13); "Latest" is 3.13.15 and flyvis requires <3.13 | Stage 0 §1, 2026-09-16 |
| Pin behaviour | "Disconnect and delete runtime" resets the pin; "Restart session" keeps it | verified 2026-09-16 |
| Editable install | `pip install -e` needs one "Restart session" before `import flyvis` works in that kernel | verified 2026-09-16, again 2026-09-17 |
| T4 High-RAM | **1.27 CU/h**, 2 vCPU, 15 GB | Resources panel |
| L4 | **1.54 CU/h**, 12 vCPU, 22.5 GB | Resources panel, 2026-09-19 |
| A100-SXM4-80GB High-RAM | **6.77 CU/h**, 12 vCPU, 80 GB | Resources panel, 2026-09-19 |
| CPU runtime | ~0.08 CU/h | Stage 0 §1 |
| Starting balance | 289.97 CU (expire 90 days from purchase) | subscription page |
| Balance 2026-09-19 after all benchmarks | ~285 CU | Resources panel |
| Sintel | 5,627,783,629 bytes over plain HTTP; 15–25 min per fresh VM; VM-local, lost on runtime change | §8 runs |
| Drive | project folder "19 - FruitFly" (id 1EMFHFcWOqn-w5Qq1i-fHVRUVzIMaazc5) in Shared Drive "Almo-2000-Research&Development" | — |

The notebook's `CU_PER_HOUR` table (T4 1.76, L4 4.00, A100 15.0) is wrong for all three GPUs; the
measured rates above supersede it.

## 2026-09-16 · Stage 0 on CPU, then the audit

**Stage 0 (`00_stage0_setup_and_benchmark.ipynb`, run on a CPU runtime):**

- (b) Premise reproduced. Min validation loss over the 50 released models: **mean 5.3143, SD
  0.0752, CV 1.42 %**. All 50 record `seed: 0`; models 000 and 007 are byte-identical, so the
  released ensemble is a reference distribution, never a baseline arm.
- (c) Standalone parameterization builds on the real connectome: 45,669 neurons, 1,513,231
  edges, 65 types; 604 (source, target) sign/magnitude groups; baseline 734 trainable, constrained
  929 (+26.6 %); parameter-matched filler 195.
- Four source types emit both signs in the connectome: Am, C2, C3, R8. Dale's law with the
  published transmitters flips **10 of 604 groups (8,590 edges)**: R8→Mi1/Mi4/Mi9/Mi15,
  Am→L1/T1, C2→L1/L2, C3→L1/L2. None targets T4/T5.

**Audit of the code (58-agent workflow, 64 min): 34 distinct issues, 27 confirmed, 7 rejected.**
Every critical one was reproduced by execution against flyvis 1.2.0:

| | Issue | Effect |
|---|---|---|
| I1 | `_dale_override` flag swallowed by `Value(**kwargs)` | arm B signs identical to arm A — silent |
| I2 | CfC gates created lazily after the optimizer was built | gates never trained — silent |
| I3 | gates per neuron (137,007) not per type (195) | accounting off ~700× |
| I4 | `SoftplusValue` stored flyvis's init as the raw value | magnitudes ~0.69 instead of ~0.005 → NaN |
| I5 | `clamp: False` | `NotImplementedError` in `Network.clamp` |
| I6 | softplus saved the leaf; flyvis steps in place before the penalty backward | `RuntimeError` on step 1 |

Put together, **the two arms were the same model**; Stage 2 would have produced a null by
construction. Also: `verify_seam` had no live checks (I7); arm-B checkpoints unloadable (I8);
§9 timed stock flyvis, passed 20 as `overfit`, and raised `TypeError` before timing (I11, I12,
I14); the §9 gate trusted a Drive flag and a CPU-only feasibility record (I15, I16).

Design-level gaps from the critic (still open, decisions for the user):

- **Arm C is a no-op**: "Dale on unambiguous types only" gives signs identical to arm A on this
  connectome, because every unambiguous type already emits one sign.
- **The filler definition contradicts itself** in PREREGISTRATION §2: "structurally inert" vs
  "per-type offsets that sum into the existing bias". Implemented as truly inert (never read).
- **Seeding**: `task.seed` only feeds the data split; data order and augmentation use the global
  RNGs; bias draws differ between CPU and CUDA for the same seed.
- **flyvis's own resume is broken**: `MultiTaskSolver.recover()` raises `TypeError`
  (`resolve_checkpoints` signature). A project wrapper is required.
- **M2 has never been computed on a real model**; GO criterion 1 (mechanism CV ≥ 5× task CV)
  is unevaluated.
- **Steady state is not converged at flyvis's `t_pre`** and how far off depends on the arm
  (stock 2.6e-4, arm A 9.4e-4, arm B with gates 8.8e-3 residual at 0.5 s).

## 2026-09-17 · Seam rewritten, verified on T4, cost measured

**`src/flyvis_seam.py` rewritten** (module-level adapters `SoftplusValue`, `DaleSynapseSign`,
`CfCGate`, `InertFiller`, `ExactExponentialDynamics`, `CfCDynamics`; gates are flyvis node
parameters; `clamp: None`; inverse-softplus init; `.clone()` for saved leaves; τ projected to
[2 ms, 1000 s]; `flyvis_network_kwargs` / `apply_arm_to_solver_config` merge onto flyvis's
defaults; `verify_seam` builds real networks and checks 22 properties on tensors).

- `verify_seam(extent=15)`: **22/22 on CPU locally and on the T4 (CUDA)**. Trainable
  A/dale_only/B = 929/929/929 (stock 734); Dale flips exactly the 10 planned groups; one flyvis
  step equals the closed form to 9e-8; zero gates == no gates bit-identically; filler inert;
  strict state-dict round trip.
- Mutation test: 8 audited bugs reinserted one at a time; `verify_seam` caught 8/8.
- Both arms through the real `MultiTaskSolver.train()` on mock Sintel: finite, gates move,
  NetworkView reload from the saved YAML equals the best checkpoint, A vs B differ in 10 sign
  groups.
- `dale_typed_cfc.py`: K3 (CPU generator on CUDA), I9 (tensor-level treatment checks), I23
  (flyvis's per-pair init; matches the seam to 1e-6), I25 fixed. `stability.py`: I28 fixed.
- Notebook regenerated with I11–I16, I29–I34 fixes; ran end to end on the T4.

**Cost per run (§9, real Sintel, T4, 24 timed steps):**

| | Arm A | Arm B |
|---|---|---|
| s/step | 0.32 | 0.33 |
| checkpoint (full validation) | 2.1 s × 71 per full run | same |
| peak GPU memory | 1.54 GB | 1.77 GB |
| full 250,000-step run | 22.5 h | 22.7 h |
| at 1.27 CU/h | ≈ 28.6 CU | ≈ 28.8 CU |

242 usable CU buys **≈ 8 full runs**; the pre-registered design needs ≈ 102 (36 seeds × 2 arms
+ 30 determinism-floor runs). Kill criterion 4 (underpowered by budget) applies as things stand.

## 2026-09-19 · Where a step goes, and what makes it cheaper

Scripts: `bench/bench_gpu.py` (profile / parallel / compile), `bench/bench_phases.py`. Mock
Sintel (step cost does not depend on image content), arm B, extent 15, batch 4, 19 frames.

**The step is GPU-bound and the GPU is saturated.** T4 profile: GPU busy **82 %** of wall,
**12,455 kernel launches per step**, dominated by gather / scatter / elementwise passes over the
1.5-million-edge tensors.

| Phase (seconds) | T4 | L4 | A100 |
|---|---|---|---|
| data loading (no workers) | 0.043 | 0.024 | 0.024 |
| network forward | 0.053 | 0.030 | **0.029** |
| backward | 0.097 | 0.055 | 0.028 |
| decoder + loss | 0.009 | 0.004 | 0.004 |
| optimizer step + activity penalty (a second backward) | 0.085 | 0.052 | — |
| **total per iteration** | **0.287** | **0.165** | **0.114** |

| Lever | Measured | Verdict |
|---|---|---|
| L4 instead of T4 | 1.7× faster, 1.21× the price | worth it |
| A100 instead of L4 | 1.45× faster, 4.4× the price; forward pass no faster than L4 | not worth it |
| concurrent seeds on one GPU (total throughput) | T4 ×4: 1.12× · L4 ×4: 1.53× · A100 ×8: 3.47× | worth it on L4 |
| `torch.compile` on the 19-frame rollout | 0.063 → 0.059 s (≈ 1.5 % of a step); tracing 129 s | not worth it |

Cost per full run: T4 single ≈ 28 CU · L4 single ≈ 24 · **L4 with 4 concurrent seeds ≈ 15** ·
A100 single ≈ 83 · A100 with 8 concurrent ≈ 24. Best setup buys ≈ 16 runs; the remaining
≈ 6× had to come from training length. Whole benchmark round ≈ 2 CU.

## 2026-09-19 · Learning curve: shortening does not work at flyvis's recipe

`bench/bench_curve.py both 20000`, real Sintel, L4, both arms concurrently, flyvis's Stage-2
configuration with `n_iters=20000` (so the 10-step learning-rate decay 5e-5 → 5e-6 is compressed
into 20k) and a validation checkpoint every 300 iterations. Stopped at ≈ 10.5k. Series in
`bench/results/curve_L4_2026-09-19.json`.

| | Arm A | Arm B |
|---|---|---|
| iteration 0 | 1212.56 | 1212.56 |
| best | 1206.05 (6,011) | 1206.10 (6,611) |
| last 10 checkpoints | 1207.7 ± 1.0 | 1207.2 ± 0.7 |
| drop over 10.5k steps | **0.40 %** | **0.44 %** |
| trend after step 300 | −0.14 / 1k steps | −0.28 / 1k steps |
| s/step, both concurrent | 0.29 | 0.29 |

flyvis's loss is the root of the **summed** squared flow error over 19 × 2 × 721 values. The
released models' stored "5.31" is the same loss on a different normalization, not a target on
this scale: evaluated with 1.2.0's own code, fully trained models score **1103–1147** (see the
evaluation two sections down). So the whole learnable range is 1212.6 → ≈ 1124, about **7 %**,
and 10.5k steps covered ≈ 6 % of it. Arm B − arm A over the last 10 checkpoints = −0.5
(t = 1.4): no difference, but vacuous as a Stage-1 gate because so little was learned.

*(An earlier version of this entry read the stored 5.31 as a 230× gap. That was wrong; corrected
2026-09-19 after the evaluation below.)*

**Cost so far this day:** ≈ 2 CU benchmarks + ≈ 1.3 CU learning curve + Sintel downloads.

## 2026-09-19 · Stock-flyvis control: the plateau is flyvis's recipe, not ours

`bench/bench_curve.py control 10500` — two **unmodified** flyvis runs (PPNeuronIGRSynapses,
clamp non_negative, 734 free parameters, no seam) on the L4 at once, real Sintel, stopped at
10,512 iterations: `stock_sched20k` (learning-rate schedule compressed to 20k, exactly what our
arms got) and `stock_sched250k` (flyvis's real 250k schedule). Series in
`bench/results/control_stock_L4_2026-09-19.json`. 53 min, ≈ 1.4 CU.

| series | start | best | last 10 checkpoints | drop over 10.5k | trend after 300 |
|---|---|---|---|---|---|
| ours, arm A | 1212.56 | 1206.05 | 1207.9 ± 1.0 | 0.39 % | −0.14 / 1k |
| ours, arm B | 1212.56 | 1206.10 | 1207.2 ± 0.7 | 0.44 % | −0.28 / 1k |
| **stock, 20k schedule** | 1212.56 | 1205.67 | 1207.0 ± 0.4 | 0.46 % | −0.19 / 1k |
| **stock, 250k schedule** | 1212.56 | 1205.31 | 1206.4 ± 0.4 | 0.50 % | −0.16 / 1k |

All four curves are the same curve. Differences over the last ten checkpoints: stock's real
schedule is 0.57 lower than the compressed one (t = 3.2; the early rate cuts cost a little);
our arm A is 0.83 above stock on the same schedule (t = 2.4; the exact integrator changes the
early dynamics slightly); arm B equals stock (+0.16, t = 0.6). All of it is within 0.1 % of the
loss, whose whole learnable range is ≈ 7 % (next section).

**Conclusion.** The slow first 10,000 steps belong to flyvis's training recipe, not to the
Dale/CfC/softplus parameterization. Shortening the run is not available with that recipe, for
stock or for us. The next section tests whether the learning rate is the reason — it is not.

## 2026-09-19 · Learning-rate sweep, and what "trained" actually scores

**Sweep** (`bench/bench_curve.py lrsweep 10500 2.5e-4,5e-4,1e-3`; stock flyvis, real Sintel,
three runs concurrently on the L4 at 0.437 s/step, flyvis's 250k schedule shape with the start
rate scaled and the 10× decay kept; series in `bench/results/lrsweep_stock_L4_2026-09-19.json`;
77 min, ≈ 2 CU):

| start rate | last 10 checkpoints | best | checkpoint noise SD |
|---|---|---|---|
| 5e-5 (flyvis default; control above) | 1206.4 ± 0.4 | 1205.3 | 0.8 |
| 2.5e-4 | 1206.6 ± 0.6 | 1203.8 | 1.1 |
| 5e-4 | 1209.1 ± 5.2 | 1204.9 | 2.8 |
| 1e-3 | 1207.7 ± 3.5 | 1200.4 | 2.1 |

A 5–20× learning rate changes nothing but the noise. **Step size is not the lever.**

**Evaluation of the released, fully trained models with 1.2.0's own loss**
(`bench/bench_eval_released.py`, eval mode, augmentation off, 1.2.0 default validation set;
`bench/results/eval_released_L4_2026-09-19.json`):

| released model | 1.2.0 validation loss | stored `validation/loss` | ratio |
|---|---|---|---|
| 000 | 1103.5 | 5.137 | 214.8 |
| 001 | 1114.8 | 5.191 | 214.7 |
| 002 | 1120.8 | 5.220 | 214.7 |
| 003 | 1132.8 | 5.247 | 215.9 |
| 004 | 1146.7 | 5.251 | 218.4 |

The stored number is the same loss under a different normalization (ratio ≈ 215, ranking
preserved), not a target on the 1.2.0 scale. On the 1.2.0 scale the picture is:

| | loss | per-element RMS flow error |
|---|---|---|
| any arm at initialization | 1212.6 | 7.33 |
| ours / stock after 10.5k steps | ≈ 1207 | 7.29 |
| released models after 250k steps | 1124 ± 17 | 6.79 |

So a full flyvis training moves the loss by ≈ 7 %, and 10,500 steps (4 % of the schedule) cover
≈ 6 % of that. The descent must accelerate later in the 250k schedule, but where is unknown
without a longer run.

**What this changes:**

- The compute problem stands, and the "shorten training" lever is closed: not by recipe,
  not by learning rate. The remaining levers are the endpoint (does the M2 fingerprint settle
  before the loss does?), the design (fewer seeds / larger effect), or more compute.
- Two pre-registered thresholds were written on the stored scale and must be restated on the
  measured one: the H2 equivalence margin ±0.08 (stored) ≙ ±17 on the 1.2.0 scale, which is
  ≈ 19 % of the entire learnable range; and the Stage-1 gate "arm B within 10 % of arm A" ≙ 112
  units, larger than the whole learnable range — it can never fail. Both need amendments.
- The premise number itself is fine: the released models' 4 % spread in this loss (1103–1147)
  is the task-error stability the project rests on.

**Cost of the day:** ≈ 2 CU benchmarks + ≈ 1.3 CU curve + ≈ 1.4 CU control + ≈ 2 CU sweep and
evaluation + four Sintel downloads ≈ 7.5 CU; balance ≈ 278 CU.

## 2026-09-19 · M2 on real models: the mechanism scatters 41× more than the loss

**Question (Stage 0 (e), kill criterion 1):** on fully trained models, is the across-seed
dispersion of the pre-registered mechanism fingerprint large relative to the dispersion of task
error? Measured on the 50 released flyvis models (`flow/0000/000..049`, 250k steps each), with
the exact pre-registered M2 definition (`stability.PRIMARY_ANALYSIS`): signed time-integrated
currents from every presynaptic type into the central T4c cell, moving edge at intensity 1 in
PD (90°) and ND (270°), window [0, 1] s after onset, dt 1/200, jointly L1-normalized → a
24-dimensional fingerprint per model. Script `bench/m2_released.py`; data
`bench/results/m2_released_local_2026-09-19.json`. Ran on the local CPU: 0 CU.

**Result — GO criterion 1 passes by a wide margin.**

| quantity | value |
|---|---|
| mechanism CV = √(total variance) / ‖centroid‖, 50 models | **0.587** (bootstrap 95 % CI 0.495–0.674; ddof = 1: 0.593; leave-one-out range 0.558–0.593) |
| task CV (stored validation loss, same 50 models) | 0.0142 (on the 1.2.0 loss scale, 5 models: 0.015) |
| ratio mechanism / task | **41.5×** (pre-registered GO threshold: ≥ 5×) |
| distance to centroid | mean 0.175; quartiles 0.124 / 0.164 / 0.206; max 0.401 (model 049); centroid norm 0.319 |
| pairwise distance / cosine | mean 0.248; cosine mean 0.75, 5th percentile 0.30, min 0.08 |
| effective dimensionality | participation ratio 4.9 of 24; eigen-shares 31 / 25 / 16 / 12 / 8 / 6 % |
| where the variance sits | Mi4 23 %, Mi9 18 %, Mi1 14 %, CT1(M10) 12 %, Tm3 11 %, T4c (recurrent) 8 % |

The centroid (share of the L1 mass, PD): Mi1 +0.19, Mi4 −0.07, TmY15 −0.05, CT1(M10) −0.05,
Mi9 −0.04, Tm3 +0.04, T4c +0.02. The disagreement is qualitative, not a matter of gain: 19 of
50 models have no Mi9 input to speak of (|share| < 1 %), 11 have no Mi4, 17 have no Tm3; model
049 is driven mainly by lateral T4a input (24 % of its mass), 046 by Tm3 (22 %), 038 by an
Mi1/Mi9 pair alone, 034 by its own recurrence plus Tm3. The overall current scale is not
identified either: total unnormalized input mass 38 ± 24 A·s across models (CV 0.62, range
7–120). The premise of the study — same task error, different circuit — is therefore true of the
paper's own models under our exact metric, not just of their parameters.

**One observation for the pre-registration (no change made):** the PD and ND halves of the
fingerprint are nearly identical for every type (mean L1 distance between halves 0.025 of a
total of 1; a PD-only 12-dimensional fingerprint gives CV 0.589). Integrating the current over
the full second erases the timing that makes T4c direction-selective, so the pre-registered
M2 measures *which inputs a model uses and how strongly*, not the direction-selectivity
mechanism itself. Dispersion results above are unaffected; an amendment adding a time-resolved
or peak-based variant would have to be declared before Stage 2.

**Frozen source-type list (P2), flyvis `ReceptiveFields("T4c")` order:** C3, CT1(M10), Mi1,
Mi4, Mi9, Mi10, T4a, T4b, T4c, T5c, Tm3, TmY15 (12 types, 24 features).

**Method notes.** `bench/m2_released.py compute` renders only the two (angle, intensity)
sequences the fingerprint reads, once, simulates each model with `Network.stimulus_response`
and rebuilds the currents on the central T4c's input edges with the dynamics' own formula
(weight × activation of the source activity). Checked against flyvis's own
`moving_edge_currents` path on models 000 and 001: max absolute difference 0 (bit-identical;
`bench/results/m2_released_reference_flyvis_path_000_001.json`). Cost: 32.5 s per model on
the local CPU with 4 shards × 3 threads (all 50 in 8 min) versus 292 s per model through
flyvis's path, which materializes all 1.5 M edge currents per frame. datamate 1.0.0 cannot
write a fresh `.h5` on Windows (unlinks an open handle); the script patches it. M2 on our own
checkpoints therefore costs nothing — the 10.5k-step checkpoints from today were lost with the
VM, so the next training run should save checkpoints to Drive.

**What this changes.** Kill criterion 1 is settled in favour of the study. The question that
decides the budget — does the fingerprint settle long before the loss does? — is now cheap to
answer on any run that keeps its checkpoints.

**Cost of the day (updated):** ≈ 7.5 CU on Colab + 0 CU for M2; balance ≈ 278 CU.

## 2026-09-19 · P5: how many seeds the primary test needs, on the measured geometry

**Rule (PREREGISTRATION §6.3):** `N_dispersion` is the smallest N per arm at which simulating
the exact pipeline — fingerprints → Euclidean distance matrix → PERMDISP(centroid) → the §3.1
decision (p < 0.05 *and* arm B less dispersed) — reaches 80 % power at a 2× dispersion ratio
(V_A / V_B of the L1-normalized fingerprints), calibrated to the measured Stage-0 geometry.
Script `bench/m2_power.py`; data `bench/results/m2_power_2026-09-19.json`. Grid: N ∈
{4 … 48}, ratio ∈ {1 … 8}, 500 replicates per cell at 999 permutations, then the cells around
`N_dispersion` re-run with 2,000 replicates at the pre-registered 9,999 permutations. Three
generators, each calibrated so the *realized* variance ratio equals the requested one:

| generator | 1.5× | **2×** | 3× | 4× | 6× | power at 8 / arm, 2× · 3× · 4× |
|---|---|---|---|---|---|---|
| `prereg` — §6.3's synthetic generator at the measured p = 24, participation ratio 4.85 | > 48 | **24** (20: 0.79 · 24: 0.88 · 30: 0.95) | 10 | 8 | 6 | 0.35 · 0.75 · 0.90 |
| `gaussian` — measured mean and full covariance | 30 | **12** (10: 0.70 · 12: 0.82 · 16: 0.92) | 8 | 6 | 5 | 0.57 · 0.91 · 0.99 |
| `bootstrap` — the 50 real fingerprints, resampled | > 48 | **30** (24: 0.75 · 30: 0.83 · 36: 0.87) | 16 | 10 | 6 | 0.35 · 0.57 · 0.77 |

Size at ratio 1 is 0.01–0.04 in every cell (nominal 0.025: two-sided F plus the direction
condition). The pre-registered 36 per arm gives 0.90 power at 2× on the real cloud, 0.97 on
the synthetic one.

**Reading.** The real cloud is the hardest of the three because it is heavy-tailed (models
049 and 046 sit at 2.3× the median distance), exactly what §6.4 warned the Gaussian generator
would hide. By the letter of the §6.3 rule `N_dispersion = 24`; on the measured cloud it is
**30**. Recommendation: record 30 in the amendment log (§13), with the 24 and 12 as the
sensitivity bracket.

**What the budget buys.** At ≈ 15 CU per run (L4, four seeds concurrently) the ≈ 242 usable
CU cover ≈ 16 runs, i.e. **8 per arm with no determinism-floor runs**. At 8 per arm the test
has 35 % power against a halving of the mechanism variance and reaches 80 % only for a ≈ 4×
reduction (distance scale halved). The pre-registered design (30–36 per arm + 15 floor runs
per arm) costs ≈ 1,350–1,530 CU, 5.5–6× the budget; even the optimistic Gaussian 12 per arm
plus floor runs is ≈ 800 CU. So the honest choices are: (a) pre-register the minimum effect of
interest as ≥ 4× and run 8–10 per arm without floor runs (≈ 240–300 CU, and the floor gate
is then unmeasured); (b) make runs cheaper, which after today's curves means proving that M2
settles early (next: a long run with checkpoints); (c) buy compute. Kill criterion 4 stays in
force until one of them is chosen.

## 2026-09-19/20 · 60k-step run of both arms, checkpoints on Drive: the loss never moves

`bench/bench_curve.py long 60000 long60k_2026-09-19` on the L4: arms A and B concurrently,
flyvis's real 250k schedule (lr 5e-5 stepwise to 5e-6), stopped at 60,492 iterations
(24 % of the schedule), a full validation checkpoint every 1,008 iterations (62 per arm),
every checkpoint mirrored to Drive (`19 - FruitFly/runs/long60k_2026-09-19/long_{A,B}`),
0.29–0.30 s/step each, **5.0 h wall, ≈ 7.7 CU**. The Colab UI lost the kernel stream twice
during the run; the process kept going and the Drive `status_*.json` files were the reliable
progress signal. The on-GPU M2 pass at the end crashed on a CPU/CUDA index-device bug
(fixed in `bench/m2_released.py`); M2 was then computed locally from the Drive copy, which
Google Drive for desktop exposes at `G:/Shared drives/...` — no re-run needed.

**Validation loss (1.2.0 scale), 62 checkpoints per arm:**

| iterations | arm A | arm B |
|---|---|---|
| 0 | 1212.56 | 1212.56 |
| 1,019 | 1208.19 | 1208.24 |
| 3,000–20,000 | 1207.27 ± 0.66 | 1207.60 ± 1.22 |
| 20,000–40,000 | 1207.37 ± 1.03 | 1207.19 ± 1.08 |
| 40,000–61,000 | 1207.27 ± 0.90 | 1207.14 ± 0.93 |
| trend after 3k | +0.002 / 1k steps | −0.012 / 1k steps |
| best | 1205.33 (23,195) | 1205.27 (38,315) |

B − A over the 58 checkpoints after 3k: −0.01 ± 1.46 (t = −0.07). **From step 3,000 to
step 60,000 the validation loss is a flat line at 1207.3 ± 0.9 in both arms.** The whole
descent of a flyvis training (1212.6 → ≈ 1124 for the released models) therefore happens, if
it happens under this recipe at all, in the last 76 % of the schedule. The released model
directories ship no training history (only a final `validation/loss` scalar and no
scheduler/optimizer/penalty config), so where their descent happened cannot be read off;
their task/dataset config matches ours key for key (batch 4, 19 frames, dt 0.02, fold 1 of 4,
augmentation identical).

Mean network activity drifts upward through the run (0.565 → 0.597 in both arms), so the
activity penalty is doing something even though the loss is not.

**M2 at every checkpoint (`bench/m2_released.py checkpoints`, computed locally from the Drive
copy in 15 min, 0 CU; data `bench/results/long60k_2026-09-19/m2_long_{A,B}.json`, analysis
`bench/m2_curve_analysis.py` → `m2_curve_analysis.json`). The question was whether the
mechanism settles before the loss does. The answer is that neither moves: **after 60k steps
both arms are mechanistically still at initialization.**

| | arm A | arm B | released models (50) |
|---|---|---|---|
| fingerprint distance travelled, init → 60k | 0.034 | 0.038 | — |
| distance from the released centroid, init → 60k | 0.389 → 0.391 | 0.395 → 0.403 | mean 0.175 (spread around it) |
| cosine(drift direction, direction to released centroid) | −0.02 | −0.16 | — |
| nearest released model at 60k | 043 at 0.31 | 043 at 0.32 | pairwise mean 0.25 |
| total input current into T4c, A·s (unnormalized) | 1.27 → 1.81 | 1.25 → 1.89 | 38 (range 7–120) |
| step per checkpoint (1,008 it.), first → last 10k | 0.0013 → 0.0005 | 0.0017 → 0.0003 | — |
| ‖A − B‖ | 0.0065 at init, 0.013 at 60k | | released pairwise 0.25 |

What a trained T4c looks like versus what ours look like (share of L1 input mass, PD):
initialization has TmY15 inhibition at −0.29 and Mi1 excitation at +0.06; the released models
have TmY15 at −0.05 and Mi1 at +0.19, Mi4 at −0.07 (ours −0.01 and moving the wrong way). The
trained mechanism is a large reorganization of the inputs, and 60k steps of this recipe do not
begin it: the fingerprint drifts slowly (step size falling with the learning-rate decay), in a
direction orthogonal to where the trained models sit, and the synapses are still 20× too
weak in total. The two arms differ from each other by 0.013 — noise on this scale.

**What this means for the design.**
- The "early M2 endpoint" idea (option (b) of the P5 section, open decision 1) is closed:
  there is no mechanism to measure before the loss descends. M2 at 60k measures
  initialization plus drift.
- Every "stop at N" shortcut is closed for N ≤ 60k (24 % of the schedule, 7.7 CU for two
  arms). Whether the 1.2.0 recipe reaches the released models' loss (≈ 1124) and mechanism at
  250k at all is now the open question — it has never been shown on this account, and the
  released directories carry no training history to check against.
- The decisive next experiment is one stock-flyvis run to 250k with checkpoints on Drive
  (single on the L4 ≈ 16 h ≈ 24 CU; the resume path exists; M2 per checkpoint is free
  afterwards). It answers three things at once: does the recipe reproduce the released loss,
  where in the schedule the descent happens, and when the mechanism forms relative to it. If it
  does not reproduce, Stage 2 cannot start on this recipe regardless of budget.

**Cost:** ≈ 7.7 CU; balance ≈ 270 CU.

## 2026-09-22 · The laptop's own GPU runs a step at T4 speed, for free

The Windows machine (HP ZBook Studio G8) has an NVIDIA RTX A2000 Laptop GPU with 4 GB; a
flyvis run needs 1.5–1.8 GB. `bench_phases.py` run locally (`scratchpad/bench_local.py`,
CUDA venv `venv312cuda`, torch 2.11+cu128; arm B, batch 4, 19 frames, mock Sintel):

| phase (s) | local A2000 | T4 | L4 |
|---|---|---|---|
| data | 0.040 | 0.043 | 0.024 |
| forward | 0.077 | 0.053 | 0.030 |
| backward | 0.145 | 0.097 | 0.055 |
| step + penalty | 0.127 | 0.085 | 0.052 |
| **per iteration** | **0.356** | 0.287 | 0.165 |

A 250k run ≈ 25 h locally at **0 CU** (the GPU sat at 1.2 GHz / 17 W of 45 W during the test,
so there may be clock headroom on a performance power plan). One run at a time, unattended,
no session cap: ≈ 6–7 runs per week while the laptop stays on. Sintel (5.6 GB) is not on
the machine yet; C: has 16 GB free.

## Open decisions (owner: user)

1. How to make Stage 2 affordable now that shortening training is closed and the early-M2
   endpoint is closed too (60k run: neither loss nor mechanism moves): shrink the design
   (≥ 4× minimum effect at 8–10 per arm) or add compute. **Before either: one stock run to
   250k with checkpoints on Drive (≈ 24 CU)** to establish that the 1.2.0 recipe reproduces the
   released loss and mechanism at all, and where in the schedule they form.
1b. Restate the H2 margin (±0.08 stored ≙ ±17 on the 1.2.0 scale) and the Stage-1 gate (10 % of
   arm A ≙ 112 units, never fails) on the measured scale; amend PREREGISTRATION §13.
1c. Whether to pre-register a time-resolved M2 variant alongside the integral one (the integral
   is direction-agnostic; see the M2 section).
1d. Record `N_dispersion` in PREREGISTRATION §13: 30 per arm on the measured cloud
   (recommended), 24 by the synthetic rule as written; and choose between (a) a ≥ 4× minimum
   effect at 8–10 per arm, (b) cheaper runs via an early M2 endpoint, (c) more compute (see
   the P5 section).
2. Arm C: drop or redefine (it tests nothing on this connectome).
3. Filler: inert (as implemented) or bias offsets (as PREREGISTRATION §2 also says).
4. Activity-penalty cadence (saves ~20–25 % per step; changes semantics; both arms).
5. Whether to buy compute: even the best measured setup needs ≈ 6× from somewhere.

## Files and where things are

| Thing | Where |
|---|---|
| seam, standalone model, statistics | `src/flyvis_seam.py`, `src/dale_typed_cfc.py`, `src/stability.py` |
| Stage 0 notebook + generator | `00_stage0_setup_and_benchmark.ipynb`, `build_stage0_notebook.py`; Drive copy id 1WBSXRA58GyMaJuLdHN1g9Kxnv35w025i |
| Stage 0 record | `stage0_results.json` in the Drive folder |
| benchmark scripts | `bench/bench_gpu.py`, `bench/bench_phases.py`, `bench/bench_curve.py` |
| learning-curve data (our arms) | `bench/results/curve_L4_2026-09-19.json` |
| stock control data | `bench/results/control_stock_L4_2026-09-19.json` |
| learning-rate sweep data | `bench/results/lrsweep_stock_L4_2026-09-19.json` |
| released models under the 1.2.0 loss | `bench/results/eval_released_L4_2026-09-19.json`; script `bench/bench_eval_released.py` |
| M2 fingerprint of the 50 released models | `bench/results/m2_released_local_2026-09-19.json` (contributions, fingerprints, CV, geometry); exactness check `bench/results/m2_released_reference_flyvis_path_000_001.json`; script `bench/m2_released.py` (compute / reference / analyze) |
| P5 power simulation on the measured geometry | `bench/results/m2_power_2026-09-19.json` (every cell, confirmation runs, summary); script `bench/m2_power.py` |
| 60k-step run, both arms | `bench/results/long60k_2026-09-19/` (loss curves, M2 per checkpoint, analysis); checkpoints on Drive `runs/long60k_2026-09-19/long_{A,B}/chkpts` (62 each); runner `bench/bench_curve.py long` (+ `--resume`), M2 `bench/m2_released.py checkpoints/merge`, analysis `bench/m2_curve_analysis.py` |
| scratch benchmark notebook | `fruitfly_gpu_benchmarks.ipynb` in My Drive (id 1iTEQByGu4mmxtxGjXeyOv_nbpJ8_ugo1) — outputs cleared once; holds the 2026-09-19 runs |
| local flyvis 1.2.0 checkout + Python 3.12 venv for tests | scratchpad `flyvis_src/`, `venv312/` (session-specific; recreate if gone) |
