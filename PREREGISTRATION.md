# Pre-registration — mechanistic identifiability in connectome-constrained models

**Frozen 2026-09-15, before any Stage-2 compute has been spent.**
**Compute units spent on this project at the time of freezing: 0.**

This document exists because `src/stability.py` can compute dozens of defensible analyses —
metric ∈ {correlation, cosine, euclidean} × normalization ∈ {none, l1, l2, max} × centre ∈
{centroid, spatial median, medoid, mean-pairwise} × test ∈ {PERMDISP, Brown–Forsythe,
permutation, bootstrap CI} — and a primary endpoint that could be any of them is not a primary
endpoint. **Naming one is what makes the endpoint real.** The module's own docstring says "do
not shop between them"; this file is the thing that makes that sentence enforceable.

**Relationship to the other documents.** `PLAN.md` holds the reasoning, the budget and the
stage gates. This file holds the commitment. Where they disagree, **this file wins.** The
machine-readable copy of the tuple in §3 lives in `src/stability.py` as the module constant
`PRIMARY_ANALYSIS`; if the constant and this file ever disagree, that is a bug in one of them
and Stage 2 does not start until they match.

**Amendment policy.** This file is append-only. Changes go in the amendment log (§13) with a
UTC timestamp, the reason, and whether any Stage-2 result had been seen at the time. An
amendment made after seeing primary-endpoint data converts the analysis to exploratory, and
must be reported as exploratory.

---

## 1. Hypotheses

Let arm A be the baseline and arm B the constrained model (§2).

- **H1 (identifiability).** The across-seed dispersion of the M2 mechanism fingerprint is
  **lower** in arm B than in arm A.
- **H2 (no cost).** Arm A and arm B validation losses are **equivalent** within ±0.08
  validation-loss units.

**The claim in the title is supported only if both H1 and H2 pass.** This is an
intersection–union test: the overall claim rejects its null only when every component rejects,
so testing two co-primary endpoints at α = 0.05 each does **not** inflate the type-I error of
the conjunction. No multiplicity correction is applied and none is needed.

Any other combination is reported as what it is:

| H1 | H2 | Reported as |
|---|---|---|
| pass | pass | The claim, supported. |
| pass | fail | "Lower mechanism dispersion, but equivalence of task error not established at margin 0.08." Report the observed loss difference and its CI. **Do not widen the margin.** |
| fail | pass | A negative result, and a publishable one: *structure is not enough* — Dale plus liquid dynamics do not pin down a mechanism. This is kill criterion #3 in `PLAN.md` §7, and it generalizes the reproducibility paper rather than contradicting it. |
| fail | fail | Negative on both counts. Report and stop. |

---

## 2. Arms, and exactly what differs between them

Three trained arms. **The released 50-model flyvis ensemble is a reference distribution for
the premise only — it is never an arm** (it records `seed: 0` in every `_meta.yaml`, models
`000` and `007` are byte-identical, and it was trained to a different budget on different
hardware).

| | **A — baseline** | **B — constrained** | **C — ablation** |
|---|---|---|---|
| Connectome / task / dataset / decoder / optimizer / LR schedule / `n_iters` / batch | identical | identical | identical |
| Magnitude parameterization | **softplus** | **softplus** | **softplus** |
| Integrator | **exact exponential** | **exact exponential** | **exact exponential** |
| Sign rule | per (pre, post) pair, fixed, as shipped by flyvis | **Dale: one sign per presynaptic cell type** | **Dale on unambiguous presynaptic types only**; `R8`, `Am`, `C2`, `C3` keep flyvis's per-pair signs |
| Time constants | static learned, one per cell type | **CfC: input-dependent, gated per cell type** | CfC |
| Tying granularity | 604 / 65 / 65 | 604 / 65 / 65 | 604 / 65 / 65 |
| Trainable parameters | 734 + **195 inert filler** = **929** | **929** | **929** |
| Seeds | 1…N, explicitly set and recorded | the same 1…N | 1…20 |
| Determinism-floor sub-arm | **15 runs at one fixed seed** | **15 runs at one fixed seed** | — |

**So arm B differs from arm A in exactly two things: the Dale sign rule and the CfC time
constants.** Softplus and the exact-exponential integrator are shared substrate, in both arms,
deliberately (`PLAN.md` §1.3–§1.4). Tying granularity is identical in the pre-registered
comparison; the `syn_count` retie (2355 → 604 groups) is **OFF** and, if ever run, is a
secondary exploratory analysis (§9), not part of this comparison.

**Pre-registered parameter matching.** Arm A receives 195 inert free scalars so that all three
arms optimize 929 parameters. Without it, "arm B has lower dispersion" is confounded with "arm
B has 26.6% more parameters". The filler must be structurally inert (per-type offsets that sum
into the existing bias) and this must be verified by `parameter_report()` before the first
Stage-2 run.

---

## 3. THE PRIMARY ANALYSIS — the full tuple, and the one statistic that decides

| Element | Committed value |
|---|---|
| **Metric (M2)** | **Presynaptic-type current decomposition into T4c**: time-integrated, **signed** current contribution from each presynaptic cell type, via `MovingEdgeCurrentView.get_temporal_contributions(...)`. **Not** tuning curves. |
| **Target cell type** | **T4c**, alone. (Both premise papers cluster on T4c.) |
| **Stimulus conditions** | Moving edge at **PD = 90°** and **ND = 270°**, `intensity = 1`. Fingerprint is the concatenation over (source type × {PD, ND}). |
| **Presynaptic cell-type set** | **All** presynaptic types with at least one edge into T4c in the shipped connectome, in flyvis's own ordering, with **`hide_source_types=None`**. The explicit list is frozen in Stage 0 and committed to this file as an amendment **before** the first Stage-2 run. |
| **Normalization** | **L1, applied once, jointly over the whole (source-type × condition) vector**: `m̂ = m / ‖m‖₁`. Not per source type, not per condition. Removes overall output gain so the statistic measures mechanism, not loudness. |
| **Distance** | **Euclidean** on the L1-normalized fingerprints. (A true metric, which PERMDISP's principal-coordinate step expects; correlation distance introduces negative eigenvalues that Anderson's correction then has to absorb.) |
| **Centre** | **Centroid** in principal-coordinate space. |
| **Test** | **PERMDISP** (Anderson 2006, *Biometrics* 62:245–253), permutation of arm labels, **`n_perm = 9999`**, RNG seed recorded. |
| **Sign convention** | Currents are used **signed**. Do not rectify. |
| **Software** | `src/stability.py`, `permdisp(D, groups, center="centroid", n_perm=9999)`, at the commit recorded in §12. |

### 3.1 The decision rule

> **THE DECISION STATISTIC IS THE PERMDISP p-VALUE ON THE M2 DISTANCE MATRIX.**
>
> **H1 is supported if and only if PERMDISP p < 0.05 and arm B's mean distance to its
> centroid is smaller than arm A's.**

The direction check is read off the `group_dispersions` the same PERMDISP call already
returns. It is not a second test and consumes no additional α; it only prevents "the arms
differ" from being reported as "the constrained arm is more identifiable" when the difference
goes the other way.

One analysis. No multiplicity budget is spent anywhere else, because nothing else decides.

### 3.2 Reported always, decides never

- **Effect size:** `log(V_A / V_B)` where `V` is `ensemble_total_variance` of the M2
  fingerprints, with a **BCa bootstrap CI** over seeds (10,000 resamples, seeds resampled
  within arm). This is the number a reader should quote. **It is not the decision rule** —
  a CI that excludes 0 while PERMDISP p ≥ 0.05, or the reverse, is reported as the
  discrepancy it is, and the PERMDISP result stands.
  *Note:* `ensemble_total_variance` uses `ddof = 0`; every other variance in the module uses
  `ddof = 1`. Harmless for a ratio at equal N, but state which is reported.
- **Per-source-type decomposition** of the distance, so a reader can see which presynaptic
  channels carry the effect. Descriptive.
- **Arm means** of every quantity above.

---

## 4. Co-primary endpoint: task-error equivalence (H2)

| Element | Committed value |
|---|---|
| Quantity | Minimum validation loss per run, the same quantity flyvis writes to `validation/loss.h5` |
| Test | **TOST**, two one-sided two-sample pooled-variance *t*-tests |
| **Margin** | **±0.08 validation-loss units** |
| α | 0.05 on each side |
| Decision | H2 passes iff **both** one-sided tests reject |

**Why 0.08 and not a percentage.** 0.08 is Karaneen et al.'s own verbatim figure — "cluster
mean errors differed by less than 0.08" — i.e. the difference their paper reports as already
sufficient to reorder which cluster matches biology. Against the released ensemble (mean
5.3143, SD 0.0752) that is **1.06 across-seed SDs**. The ±5% margin an earlier draft proposed
is 0.2657 = **3.53 SDs**, at which TOST reaches 80% power with **4 seeds per arm** — it would
pass automatically and would mean nothing.

**If H2 fails, the margin does not move.** Report "equivalence not established at margin 0.08
with N = …", give the observed difference and its CI, and let the reader judge.

---

## 5. Dale signs for the four ambiguous presynaptic types

Four presynaptic cell types emit both signs in the shipped `fib25-fib19_v2.2.json`:
**`Am`, `C2`, `C3`, `R8`**. An earlier draft resolved them by majority vote over synapse
counts. That is not acceptable: `R8`'s synapse-count-weighted vote is 45.81 (−) vs 42.80 (+),
a 3.4% margin, and `Am` *reverses* between the weighted (+64 vs −31) and unweighted (2 vs 9)
policies. A coin flip would decide the sign of ~2.05% of type-level synaptic weight across 33
of 65 postsynaptic types — including L1, L2, L3, Mi1, Mi4, Mi9, Tm1, Tm2, Tm9, Tm20 and
T4a–d, which are exactly the types the biological-agreement metric scores. It could therefore
move the *mean* of that metric, not just its variance.

**Committed: signs come from published neurotransmitter assignments, via
`DaleSignResolver(explicit_signs=...)`. The vote is not used for these four types.**

| Type | Published transmitter | Postsynaptic action | **Sign** | Evidence grade |
|---|---|---|---|---|
| `C2` | GABA | inhibitory | **−1** | **A** — direct immunocytochemistry plus transcriptomics |
| `C3` | GABA | inhibitory | **−1** | **A** — same |
| `R8` | **histamine**, with acetylcholine **co-transmission** | histamine gates a chloride channel → inhibitory at this model's columnar targets | **−1** | **A** for histamine; **the co-transmission is a documented violation of the one-transmitter premise** (see below) |
| `Am` | glutamate (candidate) | glutamate acts through the GluClα chloride channel in this circuit → inhibitory | **−1** | **C** — suggestive immunolabelling, weakest of the four |

**Citations** (each read from primary source on 2026-09-15):

- **C2, C3 — GABA.** Kolodziejczyk A, Sun X, Meinertzhagen IA, Nässel DR (2008), *PLoS ONE*
  3(5):e2110, doi:10.1371/journal.pone.0002110: *"GABA is the likely transmitter of centrifugal
  neurons C2 and C3."* Corroborated by Davis FP et al. (2020), *eLife* 9:e50901,
  doi:10.7554/eLife.50901: *"the main GABAergic inputs to L1 and L2 are C2 and C3 neurons."*
- **R8 — histamine (+ ACh).** Hardie RC (1989), *Nature* 339:704–706, PMID 2472552: a
  histamine-activated **chloride** channel mediates transmission at the photoreceptor synapse
  → histamine is inhibitory. Xiao N, Xu S, Li Z-K, et al. (2023), *Nature* 623(7987):562–570,
  doi:10.1038/s41586-023-06681-6: R8 *"separates image-forming and irradiance signals by
  co-transmitting two neurotransmitters, histamine and acetylcholine"*, with histamine going
  to unicolumnar retinotopic neurons and acetylcholine to multicolumnar accessory-medulla
  neurons. Davis et al. (2020) independently report R8 expressing both histaminergic and
  cholinergic markers.
- **Am — glutamate.** Kolodziejczyk et al. (2008): *"Glutamate may thus be used as a
  transmitter in amacrine neurons for wide-field interconnections"* (vGluT immunoreactivity in
  the amacrine α-processes). Raghu SV & Borst A (2011), *PLoS ONE* 6(5):e19472, PMID 21573163,
  list amacrine among candidate glutamatergic medulla-intrinsic types. Glutamate's inhibitory
  action in this circuit via GluClα: Molina-Obando S et al. (2019), *eLife* 8:e49373,
  doi:10.7554/eLife.49373 — *"all ON responses are lost upon pharmacological block or genetic
  loss of GluClα."*

**The honest caveat, to be stated in any write-up.** R8 co-transmits, so a one-sign-per-type
constraint is **known to be biologically wrong for R8**. We assign −1 because the cholinergic
pathway Xiao et al. describe targets multicolumnar accessory-medulla neurons that are not
among this model's 65 cell types, and because the flyvis connectome's own −1 assignments for
R8 (→ R7, L1, L3, Tm5c, Tm20) are exactly the unicolumnar retinotopic targets that receive
histamine. The connectome's **+1** assignments for R8 → Mi1, Mi4, Mi9, Mi15 are **not**
explained by the published co-transmission story, and we cannot reconstruct their provenance.
This is a substantial part of why arm C exists as a funded, pre-registered arm rather than a
fallback.

**Two blocking checks before Stage 2 (both cheap, both must be committed to the repo):**

1. **Identity check.** Confirm that flyvis's cell type `Am` is the **lamina amacrine cell**. If
   it denotes a different neuron, the assignment above is wrong and must be amended here
   before use.
2. **Override audit.** Emit the `DaleSignReport` listing every (pre, post) pair whose sign the
   explicit table overrides, the synapse-count weight affected, and the modal sign that type
   already carries in the shipped connectome. Commit it.

**Sensitivity axis (pre-registered, secondary).** M3's **mean** and **variance** are reported
separately under both sign policies (explicit-signs vs. Dale-on-unambiguous-types-only, i.e.
arm B vs arm C), because a mean shift and a variance shift are different claims.

---

## 6. Sample size

### 6.1 What governs

`N_per_arm = max(N_dispersion, N_equivalence)`, subject to the budget in `PLAN.md` §4. If the
budget cannot reach that N, `PLAN.md` §4.5 lists the pre-committed responses, and **"run it
underpowered anyway" is not among them.**

### 6.2 N for the equivalence endpoint — settled

Monte-Carlo, two-sample pooled-variance TOST, margin 0.08, α = 0.05 each side, 80% power,
SD = 0.0752 (the released ensemble's across-model SD), 20,000 replicates per point. Computed
2026-09-15:

| True difference between arms | in SDs | **N per arm** |
|---|---|---|
| 0.00 | 0.00 | **16** |
| 0.02 | 0.27 | 21 |
| 0.03 | 0.40 | 29 |
| 0.04 | 0.53 | 45 |
| 0.05 | 0.66 | 79 |
| 0.06 | 0.80 | 176 |

**Pre-registered planning value: `N_equivalence = 21`** (powered for a true gap of up to 0.02
loss units). Beyond a true gap of ~0.04 the design cannot establish equivalence at any
affordable N, and that outcome is reported, not engineered away.

### 6.3 N for the dispersion endpoint — the rule, and the evidence available now

**The scalar Brown–Forsythe power table in `PLAN.md` §6.2 (80/arm at 2×, 36/arm at 3×) does
NOT govern.** It is correct for the i.i.d. normal scalar it simulates and irrelevant to a
distance-based multivariate endpoint. Quoting it for this endpoint would be the same category
error the project criticizes.

**Committed rule:** `N_dispersion` is the smallest N at which **simulating this exact pipeline**
— fingerprints → Euclidean distance matrix → PERMDISP(centroid, 9999 perms) — reaches 80%
power at a **2× dispersion ratio**, with the simulation calibrated to the *measured* Stage-0
fingerprint geometry (dimension and effective rank) of the released ensemble. That number is
recorded in the amendment log before the first Stage-2 run.

**Evidence available at freezing time** (computed 2026-09-15 on this machine; a self-contained
Gaussian generator with a pure scale difference, fingerprints in `d` dimensions with `rank`
signal directions, N per arm, Euclidean distance matrix, `permdisp(center="centroid")`,
499 permutations):

> *(Table inserted below from the run of `permdisp_size.py`; see §6.4 for what it does and
> does not license.)*

<!--SIMTABLE-->

### 6.4 What that simulation does and does not license

- It **does** establish the qualitative ordering that drives the design: PERMDISP holds
  nominal size and is far more powerful on this pipeline than the scalar Brown–Forsythe route,
  and **Brown–Forsythe on distance-to-medoid is unusable** — its size collapses toward zero
  because the medoid contributes its own exact-zero deviation, which in high dimension
  dominates Levene's denominator.
- It **does not** license quoting a specific `N` as final. The generator produces a *pure
  isotropic scale difference with no mean shift*, which is the most favourable geometry a
  dispersion test can be handed. Real M2 fingerprints will be non-Gaussian, heteroscedastic
  across source types, and the arm difference may live in a subspace. **Treat these N values
  as a lower bound on the requirement.**
- Consequently the rule in §6.3 stands: re-run the simulation against the real Stage-0
  fingerprint geometry and record the result before Stage 2.

---

## 7. The determinism-floor gate

Run **k = 15 identical-seed runs per arm** and compute the same M2 dispersion on them. This
estimates the floor: dispersion that is not attributable to the seed at all.

**Why not k = 5.** A variance estimated from n = 5 has a 95% confidence interval of
**[0.36, 8.26] × s²** — a 23-fold span — so a 25%-of-signal threshold evaluated on a point
estimate is a coin flip. At k = 15 the interval is [0.54, 2.49] × s² (4.6-fold); at k = 20 it
is [0.58, 2.13] × s² (3.7-fold). *(Chi-square, computed 2026-09-15.)*

**Committed gate — on a confidence limit, not a point estimate:**

> Let `r̂ = s²_floor / s²_across-seed`. The gate **passes** if the **one-sided 95% upper
> confidence limit** on `r` is ≤ 0.25, where `UCL = r̂ · F₀.₉₅(N−1, k−1)`.

What that costs in practice (computed 2026-09-15):

| k | N | `F₀.₉₅(N−1, k−1)` | Point estimate `r̂` needed to certify UCL ≤ 0.25 |
|---|---|---|---|
| 5 | 48 | 5.70 | ≤ 0.044 |
| 10 | 48 | 2.81 | ≤ 0.089 |
| **15** | **48** | **2.25** | **≤ 0.111** |
| 20 | 48 | 2.01 | ≤ 0.125 |

k = 15 is the pre-registered value and is costed in `PLAN.md` §4.3 (30 runs, both arms). If
the budget stretches, k = 20 is a pre-approved upgrade and does not need an amendment.

**If the gate fails**, the study is measuring nondeterminism rather than seed-to-seed
variability, and it is re-scoped around deterministic training before anything else
(`PLAN.md` §7, kill criterion #5). Record `torch.use_deterministic_algorithms` status and the
`scatter_add` nondeterminism explicitly for every run, in both arms.

---

## 8. Mandatory diagnostics — the optimizer-geometry controls

These are **not** optional supplementary material. They are the controls that determine
whether the primary result means what it appears to mean, and they are reported whatever it
says. All are per arm.

| # | Diagnostic | Pre-registered interpretation |
|---|---|---|
| **D-a** | **M2 dispersion at step 0** (untrained networks, same seeds) | If arm B's dispersion is already lower before training, the headline is an **initialization artifact and the primary result is void**. Report it in the abstract. |
| **D-b** | `‖θ_final − θ_init‖` in **semantic** space, separately for synaptic magnitude and τ | If arm B moved dramatically less, "less dispersion" may just be "moved less from a shared start". Report the ratio. |
| **D-c** | Mechanism distance from each seed to **its own initialization** | Same purpose as D-b, in mechanism rather than parameter space. |
| **D-d** | Count of synaptic magnitudes within ε of zero at the end of training | The clamp-boundary artifact (418/604 entries in the released ensemble) must not reappear through a back door. |
| **D-e** | Empirical distribution of learned **τ relative to `dt`** | 73.7% of released-ensemble (model, cell-type) τ values sit below `dt = 0.02 s`. Both arms must report where their τ actually live; the Euler-vs-exact divergence is evaluated at that distribution, not at `dt → 0`. |
| **D-f** | M2 dispersion restricted to the **186 type-pairs that are never dead** in the released ensemble (604 − 418) | A clamp-free sensitivity analysis. |

**Why these exist.** flyvis trains with Adam at lr 5e-5. Adam's update is
gradient-scale-invariant, so a *raw* step is ~lr in any parameterization, but the *semantic*
step through softplus is `lr·(1 − exp(−m))`; at flyvis's own initialization that is ~201×
slower at the median type-pair and up to 14,410× slower at the weakest, and ~20.9× slower for
τ. Both arms start from the same deterministic initialization (only `nodes_bias` is seeded).
Less movement from a shared start produces less across-seed dispersion **by construction**.
Putting softplus in both arms removes this as an *arm* confound; D-a through D-f are what
show it has not returned through the CfC gate or the Dale sign rule.

---

## 9. Everything else is secondary — the explicit list

Everything below is **secondary or exploratory**. None of it can support the headline claim,
none of it is corrected for multiplicity, and each is labelled as secondary wherever it
appears.

**Secondary metrics**

- **R1 — response fingerprint**: L1-normalized moving-edge angular tuning curves of the eight
  T4/T5 subtypes, same pipeline (distance matrix → PERMDISP). **This is an output metric, not
  a mechanism metric.** Two networks with different circuits and identical T4c tuning are
  exactly the degeneracy this project exists to detect, and this fingerprint cannot separate
  them. Report it as a *response* result, never as a mechanism result.
- **M1** — circular dispersion of T4c preferred direction (T4c primary; Holm correction if all
  eight subtypes are reported).
- **M3** — Fisher-z dispersion of the four biological-agreement correlations (FRI, T4 tuning,
  T5 tuning, DSI). Mean shift and dispersion reported separately.
- The reproducibility paper's own qualitative test: does "best cluster by task error" → "cluster
  matching biology" transfer between two independently seeded ensembles within each arm?

**Secondary tests on the primary metric**

- Brown–Forsythe on per-seed deviations from the centre, with `center="mean_pairwise"`
  (reported) and `center="medoid"` (reported **only** alongside its measured miscalibration).
- Permutation test on the log variance ratio, median-centring each arm before pooling.
- PERMDISP with `center="median"` (spatial median).

**Secondary contrasts**

- Arm C vs arm B (the Dale sign-policy sensitivity axis).
- Arm A vs the released 50-model reference ensemble. **Confounded by training budget, flyvis
  build, hardware and dataloader order** — an observation, never a result.
- The `syn_count` retie (2355 → 604), if ever run.

**Exploratory**

- Any metric × normalization × centre × test combination not listed above.
- The per-edge usage proxy `u_ij = |w_ij|·E[|h_j|]` as an alternative fingerprint.

---

## 10. Stopping and decision rules

1. **No interim analysis of the primary endpoint.** The M2 fingerprints are not computed, and
   the PERMDISP call is not run, until all planned Stage-2 runs are complete or the budget has
   stopped the study. Training-loss curves may be monitored for failure detection; that is not
   an endpoint analysis.
2. **N is fixed before the first Stage-2 run** by the rule in §6, written into §13 with a
   timestamp, and not changed afterwards.
3. **Run order is interleaved** (A seed 1, B seed 1, A seed 2, B seed 2, …) so that a session
   kill, an expiry or a budget exhaustion truncates both arms equally.
4. **If the study stops early**, analyze at the largest N for which **complete matched pairs**
   exist in both arms, report the achieved N, and report the power the §6 simulation gives at
   that N. Do not analyze ragged arms.
5. **Failed runs** (diverged, OOM, crashed) are re-run at the same seed, up to twice. A seed
   that fails three times is dropped **from both arms** and the drop is recorded.
6. **No seed is ever excluded on the basis of its outcome.** Exclusions may only be made for a
   pre-specified technical failure, recorded at the time it happens.
7. **The primary analysis is run exactly once**, from a script committed before the data
   exists.

---

## 11. Blocking prerequisites — Stage 2 does not start until these are done

| # | Prerequisite | Why it blocks |
|---|---|---|
| **P1** | **Validate `permdisp()` against `vegan::betadisper` or `skbio.stats.distance.permdisp`** on at least one dataset. | It is a from-scratch implementation that its own docstring admits has never been checked; Anderson's imaginary-axis correction is implemented as `sqrt(|real − imag|)` and implementations differ on the absolute value. It is now the **primary test**. |
| **P2** | **Freeze the presynaptic source-type list** for T4c, with `hide_source_types=None`, and commit it here as an amendment. | flyvis's `get_temporal_contributions` defaults to `hide_source_types="auto"`, which drops types by a **data-dependent** contribution threshold — the feature set would then differ between models and between arms, which is not a comparison at all. |
| **P3** | **Re-derive the 418/604 and 69.2% clamp figures independently.** | 418/604 = 0.6921, i.e. the reported fraction of entries and fraction of variance coincide to three digits. That may be a coincidence or a transcription slip. Both numbers appear in the motivation for the whole design. |
| **P4** | **Confirm flyvis's `Am` is the lamina amacrine cell** (§5). | The pre-registered sign for `Am` depends on it. |
| **P5** | **Re-run the §6.3 power simulation against the measured Stage-0 fingerprint geometry**, and record `N_dispersion`. | §6.3's rule. |
| **P6** | **Fix `parameter_report()` usage** so the 929 / 734+195 counts come from one place, and verify the filler is inert. | §2's parameter matching. |
| **P7** | **Pass the Colab checkpoint/resume drill.** | Colab Pro has no background execution; without a verified resume path a disconnect costs the ensemble, and no budget in `PLAN.md` is real. |

---

## 12. Software, data and environment freeze

| Item | Committed value |
|---|---|
| flyvis | **v1.2.0 = commit `92b3845cc426dd309a1a0e1b3890156c42e14021`**, pinned. The `NetworkDynamics`-as-`nn.Module` seam is undocumented upstream and could break on a refactor. |
| Project modules | `src/dale_typed_cfc.py`, `src/stability.py` at the commit recorded when this file is frozen. The analysis script is committed before the data exists. |
| Python | 3.11 or 3.12 (flyvis requires `>=3.9,<3.13`; Colab runtime 2026.07 ships 3.12.13). |
| Ground truth | `flyvis/utils/groundtruth_utils.py` as shipped at `92b3845`. Scored against flyvis's own `preferred_directions` dict, **not** against a degree value taken from a paper — the degree convention is not agreed across the literature. |
| Seeds | Explicitly set and recorded per run: `network.node_config.bias.seed`, `task.seed`, global torch/numpy RNG, dataloader ordering. flyvis does **not** derive a seed from `ensemble_and_network_id`. |
| Determinism | `torch.use_deterministic_algorithms` status recorded per run. |
| Reference ensemble | The 50 released checkpoints, archived locally with their sha256 digests (the download depends on a hardcoded Drive folder id and API key and has no mirror). Load with `best_checkpoint_fn_kwargs={"validation_subdir": "validation", "loss_file_name": "loss"}`. |

---

## 13. Amendment log

Append only. Every entry: UTC timestamp, what changed, why, and whether any primary-endpoint
result had been seen.

| # | UTC | Change | Reason | Primary data seen? |
|---|---|---|---|---|
| 0 | 2026-09-15 | Initial freeze. | Round-1 adversarial review closed; design v2 locked. | **No — 0 compute units spent.** |

**Entries still owed before the first Stage-2 run** (each becomes a row above):

- The frozen presynaptic source-type list for T4c (P2).
- `N_dispersion` from the re-run pipeline power simulation, with the measured fingerprint
  dimension and effective rank (P5).
- The final `N_per_arm`, derived from Stage 0's measured `CU_PER_RUN`.
- The P1 PERMDISP validation result.
- The `Am` identity confirmation (P4).
