# Mechanistic identifiability in connectome-constrained models

**Research plan, v2 — 2026-09-15.** Supersedes v1. Rewritten after a round-1 adversarial
review (`CRITIQUE_ROUND1.md`) that found the v1 design was pre-explained by an optimizer
artifact. The design changes that fix it are locked; §1.3 says what they are.

Written for a collaborator who has not seen the conversation that produced it.

**The pre-registered analysis lives in [`PREREGISTRATION.md`](PREREGISTRATION.md), not
here.** This file is the reasoning, the budget and the stage gates. That file is the
commitment: one primary statistic, one decision rule, everything else secondary. If the two
ever disagree, `PREREGISTRATION.md` wins.

### Evidence labels used throughout

| Label | Means |
|---|---|
| **[V-live]** | Verified during this writing pass (2026-09-15) by running the code here or reading the primary source here. |
| **[V-r1]** | Verified by the round-1 reviewer from flyvis source at `92b3845`, from the released checkpoints, or from the paper itself. Not re-verified in this pass. Trustworthy, but it is second-hand. |
| **[ASSUMED]** | A judgement call or a third-party number. Nobody has checked it. |
| **[UNKNOWN]** | Nobody can check it without spending compute or downloading something. |

Do not promote a label without doing the work. Round 1 caught two false "VERIFIED" claims in
v1; that is the specific failure mode this table exists to prevent.

---

## 1. The claim

> Dale's law + per-cell-type input-dependent (CfC) time constants make
> connectome-constrained models **mechanistically identifiable** across retraining runs,
> at no cost to task error.

"Mechanistically identifiable" means: retrain the same architecture from a different seed
and you recover the *same circuit mechanism*, not merely the same task error. Operationally,
the across-seed **dispersion** of a mechanism fingerprint (§6) is lower for the constrained
model than for a matched baseline, while validation loss is statistically **equivalent**.

Two halves, two different statistical objects:

| Half of the claim | Statistical object | Test | Decides? |
|---|---|---|---|
| "mechanistically identifiable" | multivariate dispersion between arms | **PERMDISP** on the M2 distance matrix | **yes — this is the decision rule** |
| "at no cost to task error" | **equivalence** of means | **TOST**, margin 0.08 validation-loss units | **yes — co-primary** |

A non-significant difference in task error is not evidence of no cost. To say "no cost" we
must pass a pre-registered equivalence test. This is stated first because it is the most
common way a claim like this gets overstated.

### 1.1 Honest novelty ledger — what flyvis already does

The round-1 reviewer checked this against flyvis source and it is narrower than v1 implied.
Get it right before writing anything for an audience.

| Component | Status |
|---|---|
| Connectome-constrained recurrent net, fixed topology (45,669 neurons / 1,513,231 edges, FIB-25/FIB-19) | **Already flyvis.** Not ours. **[V-r1]** |
| **Cell-type parameter tying** — 604 synaptic strengths per (pre-type, post-type), 65 biases, 65 τ; 734 trainable / 2,959 fixed | **Already flyvis, and it is the central claim of Lappalainen et al. 2024.** Claiming it would be misattribution. **[V-r1]** — the reviewer read the released checkpoints: `nodes_bias`(65), `nodes_time_const`(65), `edges_sign`(604), `edges_syn_count`(2355), `edges_syn_strength`(604). |
| Weight = sign × synapse-count × strength | **Already flyvis.** **[V-r1]** |
| **Fixed synapse signs** (`edges_sign` is a buffer, never trained) | **Already flyvis.** The reviewer confirmed flyvis already fixes `edges_sign`. Our change is *not* "signs become fixed"; it is *which* sign each edge gets. |
| Learnable per-cell-type τ, init 50 ms | **Already flyvis.** **[V-r1]** |
| Non-negative synaptic magnitude | **Already flyvis**, via projected gradient descent (`param.data.clamp_(0)` at the start of every `forward()`), not a reparameterization. **[V-r1]** |
| **Softplus magnitude reparameterization** | New relative to flyvis — but as of v2 it is in **both arms** (§1.3). It is therefore part of the substrate, not part of the treatment, and **cannot be claimed as a result**. |
| **Dale's law** | **NOT in flyvis.** flyvis fixes signs per *(pre, post) ordered pair*, not per presynaptic neuron; four source types emit both signs in the shipped connectome: `R8`, `Am`, `C2`, `C3` **[V-r1]**. One sign per presynaptic cell type is a genuine change of model class, and it **overrides the published connectome for those four types**. State that in any write-up, and ablate it (§5, Stage 2, arm C). |
| **Input-dependent / CfC / liquid time constants** | **New.** `grep -rni "liquid\|cfc\|ltc\|closed.form"` over `flyvis/` returns zero hits **[V-r1]**. flyvis has a static learned per-type τ in a leaky integrator solved by explicit forward Euler. |
| **Tying granularity as a controlled knob** (e.g. retying `syn_count` from 2355 `(pre, post, Δu, Δv)` groups to 604 `(pre, post)` groups) | Granularity change of a mechanism flyvis already has. **Not a new mechanism.** OFF in the pre-registered main comparison; available as a secondary ablation. |
| **A scalar mechanism-instability statistic with a powered dispersion test** | **New** relative to both premise papers, neither of which reports one. |
| **A converged, higher-order-matched topology null** | **New** relative to arXiv 2604.04033 (3 seeds, 5–10 optimization steps). Budget permitting — see §4.7, it probably is not. |

**So the increment over flyvis is: Dale + CfC, plus a measurement methodology.** Softplus is
shared substrate. Cell-type tying is inherited. Write it that way and nowhere else.

### 1.2 What we are buying with CfC: the parameter-count confound, correctly sized

v1 said the CfC gate took the model from 734 to 767 parameters (+4%), using a
`Linear(2,8)→Tanh→Linear(8,1)` gate that `src/dale_typed_cfc.py` does not implement. The
implemented `PerTypeLiquidTau` holds per-cell-type coefficients, not a shared MLP.
Correct accounting **[V-live]** — recomputed here, and consistent with the module's
`ParameterReport`:

| Arm | Magnitudes | τ block | Bias | Total |
|---|---|---|---|---|
| **A — baseline** | 604 `edges_syn_strength` | 65 `raw_tau` | 65 | **734** |
| **B — constrained** | 604 | 65 `raw_tau` + 3×65 `gate_v`/`gate_u`/`gate_b` = **260** | 65 | **929** |

**929 vs 734 is +26.6%, not +4%.** The parameter-matched filler for arm A is therefore
**195** scalars, not 33.

**[V-live] Known code defect, fix before Stage 1.** `src/dale_typed_cfc.py`'s smoke test §6
prints `864` because it computes `real_pairs + 3*real_types + real_types` by hand and drops
the 65 biases. I ran it in this pass and confirmed the `864`. Replace that hand sum with
`cell.parameter_report()` (which already exists and already has `n_bias_params`) so there is
exactly one place where this number is computed.

Extra capacity could plausibly reduce across-seed dispersion (better-conditioned landscape)
or raise it (more directions to wander in). Either way "constrained arm has lower dispersion"
would be confounded with "constrained arm has 26.6% more parameters" unless we match. The
pre-registered mitigation is the **parameter-matched filler**: arm A gets 195 additional
free scalars in a structurally inert place (per-type bias offsets that sum into the existing
bias), so both arms optimize 929 parameters. The factorial ablation (Dale on/off × CfC
on/off) at powered N costs 4× the seeds and is out of budget (§4); arm C (§5) is the one
ablation that is funded.

### 1.3 What changed after round 1, and why — read this before anything else

Round 1 found one objection that would have killed the project after the money was spent.
It is worth understanding precisely, because it is the reason this plan exists in a v2.

**The objection (science-1).** In the 50 released baseline models, **418 of 604
`edges_syn_strength` entries are exactly 0.0 in some seeds and non-zero in others**, and
those 418 mixed entries carry **69.2% of the total across-seed variance** of
`edges_syn_strength` **[V-r1]**. That bimodality is not biology. It is flyvis's clamp-based
projected gradient descent parking weights exactly on the constraint boundary, where the
gradient is zero and a synapse that reaches 0 is dead. Softplus removes the point mass **by
construction** — a softplus magnitude can never be exactly zero. So if softplus sat only in
the treatment arm, arm B would be guaranteed to delete the dominant source of baseline
dispersion *before Dale's law did any work at all*. The headline would be a fact about a
projection operator, and a referee would find it in an afternoon.

> **Numerical caution for whoever re-derives this.** 418/604 = 0.6921, i.e. the fraction of
> entries and the fraction of variance are the same number to three digits. That may be a
> coincidence or it may be a transcription slip in the round-1 analysis. **Re-derive both
> quantities independently in Stage 0 before either appears in a write-up.**

**The fix (locked).** Softplus moves to the shared substrate. **Both arms** parameterize
synaptic magnitude as `softplus(raw)`; neither arm uses `clamp_(0)`. After that change,
arm B differs from arm A in exactly:

  **(a)** the Dale sign rule — one sign per presynaptic cell type instead of per (pre, post) pair;
  **(b)** *[optional, OFF by default]* tying granularity, if the `syn_count` retie is exercised;
  **(c)** per-cell-type input-dependent (CfC) time constants.

At the pre-registered default, (b) is off and both arms use identical 604/65/65 tying, so
**the main comparison differs in (a) and (c) only.**

**The consequences, stated plainly, because they are not free:**

1. **Arm A is not stock flyvis.** It is flyvis with softplus magnitudes and the exact
   exponential integrator (§1.4). Nothing in this study is a comparison against the
   published model.
2. **Therefore the released 50-model ensemble is not a baseline arm and never was** (§4.6).
   It is a reference distribution for the premise only.
3. **A question we can no longer answer as a headline:** "does removing the clamp change
   across-seed dispersion?" We can *observe* arm A vs the released ensemble, but that
   comparison is confounded by training budget, flyvis version, hardware and dataloader
   order, so it is a secondary observation with a caveat attached, not a result.

**Four more locked changes**, each traceable to a round-1 finding:

- **Two trained arms, unconditionally** (science-5). Every `arms=1` path and the
  "pretrained ensemble can serve as the baseline" precondition is deleted from the Stage-0
  notebook and from this plan. The released ensemble's `_meta.yaml` files record `seed: 0`
  and models `000` and `007` are byte-identical **[V-r1]** — it is not a set of independent
  retraining seeds, so its dispersion is across-*something-unknown*, not across-seed. The
  budget in §4 is recomputed with two trained arms and the result is roughly half the N
  v1 advertised. It is stated without softening.
- **PERMDISP is primary; Brown–Forsythe is demoted** (science-3). See §6.2.
- **M2 is the current decomposition, not the tuning curve** (science-6). See §6.1.
- **The baseline arm gets the exact-exponential update too** (science-10). See §1.4.

### 1.4 Why the baseline arm also gets the exact-exponential integrator

flyvis integrates with explicit forward Euler and a `1/max(τ, dt)` guard **[V-r1]**. At
initialization `dt/τ = 0.02/0.05 = 0.4`, so per-step retention is **[V-live]**:

| | per-step retention | implied effective τ |
|---|---|---|
| forward Euler, `1 − dt/τ` | 0.6000 | 39.2 ms |
| exact exponential, `exp(−dt/τ)` | 0.6703 | 50.0 ms |

That is an **11.7% per-step difference**, and it means flyvis's integrator behaves at
initialization like a 39.2 ms time constant where the config says 50 ms — a 21.7%
shortening, not a negligible `O(dt²)` term. It compounds over the 19-frame BPTT window.

Worse, **73.7% of (model, cell-type) time constants in the released ensemble sit below
dt = 0.02 s** **[V-r1]**, where flyvis pins the effective τ to `dt` and the gradient with
respect to τ is identically zero; 57 of 65 cell types straddle that boundary across seeds.
Our module has no such guard (`tau_min = 1e-3`). So the arms would differ in the integrator
*precisely in the regime most trained parameters occupy*.

**Locked:** both arms use the exact exponential update, so CfC's input-dependence is the only
integrator difference. Both arms report the empirical distribution of learned τ relative to
`dt` (§6.4) so the reader can see where the parameters actually live. The `dale_typed_cfc.py`
docstring sentence "the only remaining difference is the integrator, which agrees to
O(dt²)" is **false at the operating point** and must be corrected in the module.

---

## 2. Why now

### 2.1 The reproducibility paper (the premise)

Karaneen, Schomburg & Chklovskii, *Reproducibility and model-selection stability in
connectome-constrained circuit modeling*, bioRxiv `10.64898/2026.04.18.717873`, posted
2026-04-20; a Matters-Arising-style comment on Lappalainen et al. 2024 (*Nature*).
Every quotation below is verbatim and was checked by the round-1 reviewer **[V-r1]**.

What they did: trained two new 50-network ensembles with stock flyvis, clustered models by
T4c naturalistic-stimulus activity (UMAP + Gaussian mixture), then correlated per-model
FRI / T4-tuning / T5-tuning / DSI against literature ground truth.

What they found: **task error is stable, circuit mechanisms are not.** Cluster mean task
errors "differed by less than 0.08" across both new ensembles, yet the mapping "cluster
ranked best by task error" → "cluster matching biology" did not transfer. An ad hoc rule
(take the cluster containing the most networks among the top 20% by task error) worked, but
they say it "was arrived at post hoc."

**Two sentences from that paper that any write-up must quote, or a referee will.**

1. Their Discussion asks for exactly what we propose: "more detailed characterization of
   variability across retraining runs, sensitivity of conclusions, and justification of
   ensemble size would improve interpretability."
2. They also record: "in correspondence, the authors clarified that the paper's primary
   accuracy claims rest on the median responses across the top 20% of models ranked by
   validation error, and that the clustering analysis is exploratory." **Do not frame
   Lappalainen et al. as resting on cluster selection.** They do not.

**The gap we exploit.** Their instability is demonstrated *qualitatively*. There is no
scalar instability statistic anywhere in the paper and no significance test.

### 2.2 The premise is already half-confirmed, for free

Computed on 2026-09-15 from the 50 released flyvis checkpoints
(`results/flow/0000/000..049`, 3.4 MB, no training) and independently reproduced by the
round-1 reviewer **[V-r1]**:

| Quantity | Across-model spread (N=50) |
|---|---|
| Minimum validation loss (task error) | mean **5.3143**, SD **0.0752**, CV **1.42%** |
| `nodes_bias` (65 values) | median per-element CV **20.2%** |
| `nodes_time_const` (65 values) | median per-element CV **98.3%** |
| `edges_syn_strength` (604 values) | median per-element CV **121.4%** |

Task error varies by 1.4% across models; time constants and synaptic strengths vary by
~100%. That is a 14×–85× ratio, visible for the price of a 3.4 MB download.

**Three caveats, all load-bearing:**

1. **Parameter dispersion is not mechanism dispersion.** Different parameters can implement
   the same function (`syn_strength` and `bias` trade off directly). This is exactly why
   Stage 0 must measure *mechanism*. Do not put these CVs in a paper as a result.
2. **69.2% of the `edges_syn_strength` variance is the clamp artifact** (§1.3), so the
   121.4% figure is mostly an optimizer fact, not a biological one.
3. **These 50 models all record `seed: 0`** **[V-r1]**, so this is across-*something*
   dispersion. It establishes that the models differ; it does not establish that seeds are
   what makes them differ.

### 2.3 The topology paper (the underpowered negative worth redoing)

*Topological Sensitivity in Connectome-Constrained Neural Networks*, arXiv 2604.04033
(Dhiman, 2026-04-05). Compares the empirical connectome against a naive random graph and a
degree-preserving rewired null: "Apparent topology advantages ... do not robustly persist
under degree-preserving controls." All following verbatim **[V-r1]**:

- **Three optimization seeds.** "Limited seeds. Core comparisons use three optimization
  seeds." Three seeds cannot distinguish a 4× dispersion difference from noise.
- **5 and 10 optimization steps.** "The matched-step protocol evaluates models after 5 and
  10 optimization steps ... does not address longer training trajectories or final
  convergence." flyvis trains for 250,000 iterations.
- **Aggregation.** v1 said "mean ± SD across seeds only — no variance test at all." That is
  unfair as written: they also report bootstrap CIs for control-minus-connectome differences
  and a five-sample degree-preserving ensemble. **The accurate criticism is narrower and
  survives intact: a CI on a mean difference is not a test of variances.** Say it that way.
- **Their own headline confound, which v1 omitted and which bears directly on our design:**
  training both graphs from a *shared random initialization* removed the early-loss
  advantage. We share initialization between arms too (§1.3), so we inherit the same issue
  and must report the step-0 diagnostics (§6.4) that detect it.
- They rewire the **neuron-level** graph (45,669 nodes / 1,513,231 edges, directed
  double-edge swaps, self-loops fixed, 250,000 accepted swaps, acceptance 0.992) while
  "Rewiring modifies only the source-target incidence structure while leaving the metadata
  tables and parameter assignments unchanged," keeping 734 trainable / 2,959 fixed in every
  condition.

**Interpretation (ours, not the paper's words) [ASSUMED]:** if edge rows keep their original
`(source_type, target_type, Δu, Δv)` weight-group labels while `source_index`/`target_index`
are rewired, a rewired edge carries a weight keyed to a type-pair whose types it no longer
connects. The null then preserves parameter count but scrambles the *type-consistency* of
weight sharing — simultaneously a topology null and a weight-assignment null. **Verify
against their code before citing this as a criticism.**

### 2.4 Why the two papers combine into one project

The reproducibility paper says structure-plus-task-error is not enough to pin down a
mechanism. The topology paper says (underpoweredly) that the structure might not even be
doing the work. Both are claims about *dispersion across retraining*, and neither measured a
dispersion properly. The project is: define the dispersion, power the test, and ask whether
adding biological constraints collapses it.

---

## 3. Hardware and environment

### 3.1 Local

- **NVIDIA RTX A2000 Laptop GPU, 4 GB VRAM** (Ampere, sm_86).
- **[V-live] Local Python cannot run flyvis.** The interpreter here is **Python 3.14.3**
  (confirmed this pass), outside flyvis's `requires-python = ">=3.9,<3.13"`, and the
  installed torch is a **CPU-only build** (`2.10.0+cpu`). Both `src/` modules are written to
  need neither flyvis nor CUDA, and both pass their self-tests on this interpreter
  **[V-live]** — `python src/stability.py` → ALL CHECKS PASSED in 0.80 s;
  `python src/dale_typed_cfc.py` → ALL 30 CHECKS PASSED in 0.11 s. Keep that property: local
  smoke tests must never require flyvis or CUDA.
- A local flyvis environment needs a clean Python 3.11/3.12 venv with a CUDA torch build for
  sm_86. **[UNKNOWN]** whether 4 GB is enough at the default `extent=15`. The v1 arithmetic
  estimate (~1.4 GB of stored activations for 19 timesteps at batch 4, before optimizer
  state, connectome tables and fragmentation) is arithmetic, not measurement.

### 3.2 Colab Pro — and the one property that shapes every run script

- **Colab Pro. There is no background execution** — that is Pro+ only **[ASSUMED]**,
  from Google Workspace Updates and the Colab FAQ. Consequence: **closing the browser, a
  laptop sleeping, or a flaky network can kill the run.** This is not a nuisance, it is a
  design constraint (§4.5).
- **Compute units expire 90 days after the purchase they belong to** (verbatim from the
  purchase flow: "Any compute units associated with this purchase will expire after 90
  days"). There is **no API and no UI** exposing per-grant expiry. If the 289.97 CU
  accumulated over several months, part of the balance is closer to expiry than it looks.
  **Front-load Stage 2.** If the balance were purchased today it expires 2026-12-14.
- Maximum session length on Pro is **not published** **[UNKNOWN]**. Plan for "could be cut at
  any time."
- The only sanctioned anti-idle lever, verbatim from the FAQ: "Idle timeouts only apply if
  code execution terminates." So structure each job as **one long-running cell**. Do not use
  browser auto-clickers; they are what the FAQ's prohibited-use list targets, and accounts
  get throttled.
- Call `google.colab.runtime.unassign()` the instant a job finishes. Single highest-value
  line in the Colab setup.
- **The Drive working folder is a SHARED DRIVE.** It mounts at
  `/content/drive/Shareddrives/Almo-2000-Research&Development/19 - FruitFly`, **not** under
  `MyDrive`. Any path logic must resolve this robustly and fail loudly rather than silently
  writing to a path that is not the folder.

---

## 4. Compute budget — recomputed from scratch

Everything in this section is arithmetic on four inputs. Three are known; the fourth is the
number Stage 0 exists to produce.

### 4.1 Inputs

| Input | Value | Label |
|---|---|---|
| Balance | **289.97 CU** (1 CU = $0.10 → **$29.00**) | given |
| Expiry | **90 days after the purchase each grant belongs to**; per-grant dates not exposed | **[ASSUMED]** |
| T4 rate | **1.76 CU/hr** | **[ASSUMED]** — third-party; Google publishes no rate table **[V-r1]** |
| A100 rate | **15 CU/hr** | **[ASSUMED]**, same caveat |
| Wall-clock per training run | **[UNKNOWN]** — nothing published; Stage 0 measures it | — |

**[V-live] What the balance buys:**

```
289.97 CU / 1.76 CU/hr  = 164.8 T4-hours
289.97 CU / 15   CU/hr  =  19.3 A100-hours
```

**A100 is off the table.** It would have to be **8.52× faster per run** to break even. This
model will not deliver that: 734–929 trainable parameters, batch size 4, a sequential
19-step Euler/exponential loop, and a `scatter_add_` over 1.5 M edges per timestep. It is
latency- and bandwidth-bound with no dense GEMMs to saturate tensor cores. **Rule: T4 by
default.** Switch to L4 only if a measured step-time ratio beats the measured rate ratio.

**Calibrate the rate once, in-session, before any large spend.** Note the balance in
`View resources`, run a measured 30 minutes, re-read the balance, divide. There is no
programmatic way to read the balance — the VM never sees it **[V-r1]**, so this is a manual
read-and-record step, and the recorded number overrides the 1.76 assumption everywhere.

### 4.2 Overheads and reserve

| Line | CU | Why |
|---|---|---|
| Rate calibration + Colab install smoke test | 2 | §5 Stage 1.6 |
| Checkpoint/resume drill (force a disconnect, resume from Drive) | 2 | mandatory on Pro (§4.5) |
| Reserve (15%) | 43.5 | session kills, re-runs, expiry slack |
| **Usable for training** | **242.47** | = **137.8 T4-hours** |

### 4.3 The run inventory — what "the honest design" actually costs

This is the part v1 got wrong by assuming one trained arm. Every run below costs the same
`c` CU, because every arm trains the same size model on the same task for the same number of
iterations.

| Line | Runs | Why it is not optional |
|---|---|---|
| Arm A (baseline) | **N** | D2: two trained arms unconditionally |
| Arm B (constrained) | **N** | |
| Determinism floor, arm A | **15** | D9: a variance from k=5 has a 95% CI of [0.36, 8.26]×s² **[V-live]** |
| Determinism floor, arm B | **15** | |
| Arm C (Dale on unambiguous types only) | **20** | D6: pre-registered ablation, not a fallback. Underpowered on purpose; it can only detect very large differences. |
| **Total** | **2N + 50** | |

### 4.4 The sizing table

`N_per_arm = floor( (242.47 − 50·c) / (2·c) )`, with `c = hours_per_run × 1.76`. **[V-live]**

| Wall-clock/run (T4) | `c` (CU) | **N per arm** | Total runs | CU used |
|---|---|---|---|---|
| 0.25 h | 0.44 | 250 | 550 | 242.0 |
| 0.33 h | 0.58 | 183 | 416 | 241.6 |
| 0.50 h | 0.88 | **112** | 274 | 241.1 |
| 0.60 h | 1.06 | 89 | 228 | 240.8 |
| 0.75 h | 1.32 | 66 | 182 | 240.2 |
| **1.00 h** | **1.76** | **43** | 136 | 239.4 |
| 1.50 h | 2.64 | 20 | 90 | 237.6 |
| 2.00 h | 3.52 | 9 | 68 | 239.4 |

**Break-even points [V-live]** — the largest per-run cost at which a given design still fits
inside 242.47 usable CU:

| Design | Max CU/run | Max T4-hours/run |
|---|---|---|
| Stage 2 at N=48/arm + floor(2×15) + ablation(20) | 1.661 | **0.944 h** |
| Minimal Stage 2: N=36/arm + floor(2×15), no ablation, no Stage 3 | 2.377 | **1.351 h** |
| Stage 2 at N=48 **plus Stage 3** (2 null arms at N=48) | 1.002 | **0.569 h** |

### 4.5 Does the honest design fit? — the plain answer

**It fits if and only if one training run costs less than ~0.94 T4-hours. Nobody knows
whether it does.** Wall-clock per run at the flyvis configuration is not stated in the
*Nature* paper, the README, the docs, the CHANGELOG or the source **[V-r1]**; the only
hardware hint anywhere is the CLI's default LSF queue `--q gpu_l4`. That is the entire
reason Stage 0 exists.

So the honest statement has three branches, and they are pre-committed here so that nobody
has to make this call while looking at a spend:

- **If a run costs ≤ 0.57 T4-h:** everything fits, including Stage 3. Spend the surplus on
  seeds, never on a bigger model (§4.6).
- **If a run costs 0.57–0.94 T4-h:** Stage 2 fits at N=48/arm with the floor and the
  ablation. **Stage 3 does not fit and is deferred.** Say so in the write-up rather than
  running a 3-seed topology comparison, which is the exact mistake we are criticizing.
- **If a run costs > 0.94 T4-h:** **the honest design does not fit.** Do not quietly shrink
  it. The options, in the order they should be considered:

  1. **Fewer seeds.** N=36/arm (the smallest N the scalar power table supports for a 3×
     ratio) with the full floor and no ablation fits up to 1.35 T4-h/run. Below N≈36 the
     experiment cannot support a dispersion claim at all and should not be run.
  2. **Smaller task.** Cut `task.n_iters` (both arms matched) against a Stage-0 learning
     curve; then `vertical_splits`; then `network.connectome.extent` below 15 (biggest
     lever, but it shrinks the spatial receptive field and may distort motion tuning — both
     arms must share it and the distortion must be characterized). **Never cut `batch_size`
     below 4:** that raises gradient noise, which directly inflates the dispersion we are
     measuring.
  3. **Two months.** Run Stage 2 across two CU grants. This trades against the 90-day
     expiry, so it only works if the current balance is young.
  4. **Ask the Workspace admin about Pro+.** Two benefits, both material: more units, and
     **background execution**, which removes the single largest operational risk in §4.5.

**The disconnect rule (Pro has no background execution).** Structure Stage 2 as a *queue of
independent runs* driven by one long-running cell, with a checkpoint to Drive at least every
10 minutes and at every run boundary, written atomically (`.part` then rename). A disconnect
must cost **one run's partial progress, never the ensemble.** Before spending anything on
Stage 2, run the 2-CU drill: start a job, kill the browser, reconnect, confirm it resumes
from the Drive checkpoint. If that drill fails, nothing else in this budget is real.

### 4.6 Surplus rule

**Any surplus buys MORE SEEDS, never a bigger model.** The headline is a *dispersion*, which
needs roughly 4× the N of the corresponding mean comparison, and the payoff curve is steep
exactly where we sit: at n=50/arm the Brown–Forsythe power for a 3× variance ratio on a
normal scalar is 0.942 **[V-live, measured here]**, but at a 2× ratio it takes n=80/arm to
reach 0.80 **[V-live]**. A bigger model raises `c`, which lowers N, which lowers power —
strictly the wrong direction. Pre-register the final N before looking at any result.

### 4.7 What the free reference ensemble does and does not save us

The 50 pretrained flyvis models are a 3.4 MB download (`flyvis download-pretrained`,
sha256-verified; the round-1 reviewer downloaded them and the digest matched **[V-r1]**).

**They are a reference distribution. They are not an arm.** Two independent reasons:

1. They were trained for 250,000 iterations; our arms will almost certainly train for fewer,
   on different hardware, with a different flyvis build. Comparing against them confounds
   parameterization with training budget.
2. **They are not 50 independent retraining seeds.** `_meta.yaml` records `seed: 0` and
   models `000` and `007` are byte-identical **[V-r1]**. Whatever makes them differ is
   unrecorded — plausibly CUDA nondeterminism in `scatter_add_`. Their dispersion is
   across-*unknown*, not across-seed.

What they *do* buy, for 0 CU: the premise numbers in §2.2, a harness cross-check against the
reproducibility paper's own analysis path, and the Stage-0 mechanism-dispersion measurement
that sizes the effect we are chasing. That is worth having. It is not worth ~140 CU of
retraining, and it is not a control arm.

**Two corrections to v1's description of the release** (both **[V-r1]**):

- v1 said `best_chkpt` and `chkpts/chkpt_00000` are "byte-identical". They are byte-*different*
  in all 50 models (pickle metadata) but **tensor-identical** in all 50. The substantive
  claim — one saved state per model, no training trajectories — is right; the stated evidence
  was wrong. This is exactly the kind of sentence a VERIFIED label is supposed to protect.
- The name `best_chkpt` **does not appear anywhere in flyvis v1.2.0 source**. Only
  `best_chkpt_index` does. Any code that globs for a file called `best_chkpt` is relying on
  an archive convention, not on the library. Discover checkpoints with
  `glob(<model>/chkpts/chkpt_*)` instead.

**Gotcha that will otherwise cost an hour:** `Ensemble.__init__` defaults to
`best_checkpoint_fn_kwargs={"validation_subdir": "validation", "loss_file_name": "epe"}`
**[V-r1]** but the shipped archive contains `validation/loss.h5`. Always pass
`loss_file_name="loss"`.

---

## 5. Stages

### Stage 0 — Establish the premise and produce the cost number

**Purpose:** confirm the premise on *mechanism* rather than parameters, retire the largest
technical risk, and produce the one number that sizes everything else.

The Stage-0 notebook (`00_stage0_setup_and_benchmark.ipynb`) runs on Colab and **does spend
compute units** — two cells are tagged `[!! BURNS COMPUTE UNITS !!]`. The claim in v1's
README that Stage 0 is "local and free" was wrong.

- **0.1 Environment.** flyvis v1.2.0 (= commit `92b3845` **[V-r1]**). Colab runtime 2026.07
  ships Python 3.12.13, inside flyvis's supported range **[V-r1]**. Do **not** use the
  `[pretrained]` extra: it pins `numba==0.55.2` **[V-r1]**. Set `FLYVIS_ROOT_DIR` *before*
  importing flyvis — the package computes `results_dir`/`sintel_dir` at import time and calls
  `torch.set_default_device(...)`, mutating the global torch default device for the process
  **[V-r1]**.
- **0.2 Reference distribution.** `flyvis download-pretrained` (3.4 MB). Reproduce the
  reproducibility paper's four violins (FRI, T4 tuning, T5 tuning, DSI over the top 20% by
  validation error) so we know our harness agrees with theirs.
- **0.3 Define and run the metrics.** M1, M2, M3 (§6) on the reference ensemble. Report,
  with bootstrap CIs, the ratio of **mechanism** dispersion to task-error dispersion. The
  parameter-space version is already known (§2.2); the mechanism-space version is the one
  that counts, and it is what the GO gate reads.
- **0.4 Feasibility, not just cost.** Instantiate a real `flyvis.network.Network` and check
  that `DaleSign` / `SynapseMagnitudeSoftplus` / `CfCDynamics` construct against the real
  `ConnectomeFromAvgFilters`; that `edges_sign` / `edges_syn_strength` / `nodes_time_const`
  have the shapes the modules assume (604 / 604 / 65); that the CfC module satisfies flyvis's
  `NetworkDynamics` interface; and that a re-parameterized network takes one optimizer step
  through `MultiTaskSolver`. **[V-r1] this was missing from the v1 notebook entirely** — a
  paid session would have answered "what does it cost" without answering "is it possible."
- **0.5 Seeding.** **[V-r1]** flyvis does not derive a seed from `ensemble_and_network_id`;
  that string only names a directory. We must write and record explicit per-seed overrides:
  at minimum `network.node_config.bias.seed` and `task.seed`, plus global torch/numpy RNG and
  dataloader ordering. Record `torch.use_deterministic_algorithms` status per run.
- **0.6 Cost probe.** Measure seconds per training iteration at the Stage-2 configuration and
  peak GPU memory, and fit a learning curve to justify a reduced `n_iters`. Two accuracy
  requirements the round-1 reviewer identified **[V-r1]**:
  - `task.n_iters` is not an iteration count. `MultiTaskSolver.train` computes
    `n_epochs = ceil((n_iters − iteration)/len(dataloader))` and runs complete epochs; at the
    shipped defaults `len(train_data) == 12`, so `n_iters=100` runs 108 iterations and
    `n_iters=300` runs 300. **Fit on `solver.iteration`, not on the requested counts**, or
    the per-iteration cost is biased ~4% low.
  - Validation cost is under-counted. `chkpt_every_epoch: 300` means short probe runs
    checkpoint twice, so validation lands entirely in fixed overhead and is never multiplied
    up; a real 250k-iteration run does ~70 checkpoints. The bias is small (<1%) and absorbed
    by the overhead margin, but it is an **under**-estimate and the notebook must say so.

**Deliverables:** `stage0_results.json` with `CU_PER_RUN` and every input to it; mechanism
dispersion (M1/M2/M3) on the reference ensemble; the feasibility checklist from 0.4.

**GO/NO-GO gate.** GO requires all of:

1. Mechanism dispersion across the 50 released models is large relative to task-error
   dispersion: **mechanism CV ≥ 5 × task-error CV**.
2. `CU_PER_RUN` is consistent with **N ≥ 36/arm** after §4.3's full inventory — i.e.
   ≤ 1.35 T4-hours/run for the minimal design, ≤ 0.94 for the design with the ablation arm.
3. The Stage-0.4 feasibility checks pass: a re-parameterized network trains.

The determinism-floor gate is **not** here. It cannot be evaluated until the floor sub-arms
run in Stage 2 (§5, Stage 2; the statistics are in `PREREGISTRATION.md` §7).

### Stage 1 — Build and check the constrained model (local, ~4 CU on Colab)

The implementation seams are **[V-r1]**: flyvis dispatches by class *name* through
`forward_subclass` → `find_subclass`, which walks `cls.__subclasses__()` recursively, so a
subclass defined in our module is discoverable as soon as the module is imported. No fork, no
registry edit, no changes to connectome / task / dataset / solver. Use the **Python API**
(`Network(...)` / `MultiTaskSolver(config=...)`), not the Hydra CLI: `find_subclass` only
sees subclasses of already-imported modules and there is no plugin hook in
`flyvis_cli/training/train_single.py` **[V-r1]**.

- **1.1 Shared substrate (both arms).** Softplus magnitudes on the 604-vector (≈2,500×
  cheaper than per-edge **[V-live]**, the module reports 2,505.3×); `clamp: "non_negative"`
  removed — the flyvis base class explicitly warns that a reparameterized parameter must not
  also be clamped **[V-r1]**. Exact-exponential integrator in both arms (§1.4).
- **1.2 Dale sign (arm B only).** A `Parameter` subclass grouping by `["source_type"]` only
  (≈63 groups instead of 604 — record the exact count, since not every cell type emits edges
  inside the modeled connectome). Signs for the four mixed-sign types come from
  `DaleSignResolver(explicit_signs=...)` populated from published neurotransmitter
  assignments, **not** from a vote. The table and its citations are in
  `PREREGISTRATION.md` §5; the reasoning is §5.1 below.
- **1.3 CfC dynamics (arm B only).** A `NetworkDynamics` subclass that also subclasses
  `nn.Module`, so gate parameters register as the child module `dynamics`. **[V-r1]**
  verified end-to-end (appears in `named_parameters()`, reaches the optimizer param group and
  the checkpoint `state_dict`, gradients flow), but **undocumented and untested upstream** —
  pin `flyvis==1.2.0`.
- **1.4 Parameter-matched filler (arm A only).** 195 inert scalars (§1.2).
- **1.5 Correctness checks.** `parameter_report()` counts (929 / 734+195); gradient flow;
  checkpoint save/load round-trip; determinism under fixed seed on CPU; and a **measured
  throughput penalty vs arm A**. Expect a real one: an input-dependent τ evaluates a gate on
  `(batch, 45669)` node states at *every* step, forfeiting the per-forward `AutoDeref` cache
  that amortizes parameter dereferencing across timesteps. If CfC makes runs ≥2× more
  expensive it halves N, and the cheaper per-cell-type gate must be used instead.
- **1.6 Colab smoke test + rate calibration + resume drill** (~4 CU, §4.2).

**GO/NO-GO gate.** On a matched short run (same iterations, same data, same decoder), arm B's
median validation loss must be **within 10% of arm A's median**. Pre-registered now. If it
fails, apply kill criterion #2 (§7).

### Stage 2 — The identifiability experiment (Colab, main spend)

**Design.** The full arm table, the analysis tuple and the decision rule are in
`PREREGISTRATION.md`. Summary:

| | Arm A — baseline | Arm B — constrained | Arm C — ablation |
|---|---|---|---|
| Topology | real connectome | real connectome | real connectome |
| Magnitude | **softplus** (shared) | **softplus** (shared) | **softplus** (shared) |
| Integrator | **exact exponential** (shared) | exact exponential | exact exponential |
| Signs | per (pre, post) pair, fixed (flyvis) | **Dale: one sign per presynaptic type** | **Dale on unambiguous types only**; per-pair signs kept for `R8`, `Am`, `C2`, `C3` |
| τ | static learned per type | **CfC, input-dependent per type** | CfC |
| Tying | 604 / 65 / 65 | 604 / 65 / 65 (knob OFF) | 604 / 65 / 65 |
| Parameters | 734 + **195 inert filler** = 929 | 929 | 929 |
| Task, data, decoder, optimizer, schedule, `n_iters` | identical | identical | identical |
| Seeds | 1..N, explicit and recorded | the same 1..N | 1..20 |
| Determinism floor | **15 runs at one fixed seed** | **15 runs at one fixed seed** | — |

**Primary endpoint and decision rule:** see `PREREGISTRATION.md` §3. One statistic decides.

**Run order:** interleave arms (A seed 1, B seed 1, A seed 2, …) so that a budget death or a
session kill truncates both arms equally. Analyze only complete matched pairs.

**Operational requirements:** one long-running cell per job; a queue of runs, not one giant
run; atomic local checkpoint then Drive mirror as `.part` + rename; time-based checkpointing
(every ~10 min) as well as step-based; `runtime.unassign()` on completion; `FLYVIS_ROOT_DIR`
on local disk (not drivefs — many small HDF5 writes over FUSE are pathologically slow and
would contaminate timing), with only checkpoints and results mirrored to Drive.

**GO/NO-GO:** if the dispersion result is null at the pre-registered N, that is a **result**,
not a failure — kill criterion #3, and the paper becomes "structure is not enough."

### Stage 3 — The topology test, done properly (deferred unless runs are cheap)

**[V-live]** Stage 3 fits only if a run costs ≤ 0.57 T4-hours (§4.4). Otherwise it is
deferred and said to be deferred. Running it at 3 seeds would repeat exactly the mistake we
criticize.

If it runs: same convergence budget as Stage 2, the powered N, a dispersion test, and a
**higher-order-matched** null rather than only a degree-preserving one. Arms: (1) real
connectome, reused from Stage 2 for free; (2) degree-preserving null (directed double-edge
swaps, matching arXiv 2604.04033 so the comparison is like-for-like); (3) higher-order-matched
null preserving degree *plus* reciprocity and the directed 3-node motif spectrum — because a
degree-preserving swap destroys motif structure, so "degree-preserving null performs equally"
is consistent with *motifs* carrying the signal.

**Rewire at which level?** (§2.3.) **Neuron-level** (what arXiv 2604.04033 did) is jointly a
topology and a weight-assignment null. **Type-level** — rewire the 65-node cell-type filter
graph, then expand through flyvis's normal `add_conv_edges` machinery — preserves
type-consistent weight sharing and keeps 604 magnitudes by construction. **Run type-level if
only one fits**, and state the consequence. Preserving the hexagonal `(Δu, Δv)` offset
structure under rewiring is the main implementation risk; budget local time for it.

**Endpoints:** the same M1/M2/M3 machinery, plus mean task error (here a *difference* test is
appropriate — we are asking whether topology helps, not whether it is equivalent), plus the
per-edge usage proxy `u_ij = |w_ij| · E[|h_j|]` from the topology paper, which vectorizes
directly into M2 as an alternative fingerprint.

### Stage 4 — Scale to MaleCNS (optional, out of budget, do not start)

*Sexual dimorphism in the complete Drosophila male central nervous system connectome.*
Two versions, and **they do not agree, so cite the one you use** **[V-r1]**:

| | bioRxiv `10.1101/2025.10.09.680999` v2 (2025-10-30) | *Cell* (2026), article `S0092-8674(26)00942-6` |
|---|---|---|
| Neurons | 166,691 | 166,700 **[ASSUMED — not accessible to verify]** |
| Types | 11,691 | 11,710 **[ASSUMED]** |
| Isomorphic / dimorphic / male-specific / female-specific types | 7,205 / 114 / 262 / 69 | v1 of this plan quoted 8,069 / 138 / 289 / 71 with no version label; those numbers are **[UNKNOWN]** and must not be reused until checked |
| ~125 M connections, 44 person-years of proofreading | — | **[ASSUMED]** |

Central brain, optic lobes and ventral nerve cord imaged as one volume through an intact neck
connective, enabling complete sensory-to-motor circuits.

Why it is Stage 4 and explicitly optional: flyvis's connectome loader
(`ConnectomeFromAvgFilters`) assumes a hexagonal retinotopic lattice with `(Δu, Δv)`
offset-convolutional weight sharing, and **that assumption does not hold outside the optic
lobe** — a MaleCNS model needs a new `Connectome` implementation *and* a new tying scheme.
There is also no task: Sintel optic flow is an optic-lobe task. And compute is out of scale.
**Do not start Stage 4 until Stages 2 and 3 have produced a result.**

---

## 5.1 Dale signs: from a coin flip to a citation (D6 / science-9)

v1 resolved the four mixed-sign presynaptic types by majority vote over the shipped
connectome. Round 1 showed why that is not acceptable **[V-r1]**: `R8`'s synapse-count-weighted
vote is **45.81 (−) vs 42.80 (+)**, a 3.4% margin, and `Am` *reverses* between the weighted
(+64 vs −31) and unweighted (2 vs 9) policies. A coin flip therefore decides the sign of
**2.05% of type-level synaptic weight across 33 of 65 postsynaptic types**, including L1, L2,
L3, Mi1, Mi4, Mi9, Tm1, Tm2, Tm9, Tm20 and T4a–d — precisely the types that carry non-zero
polarity in the FRI ground truth and the binary DSI target that M3 scores. The vote can
therefore shift the *mean* of the biological-agreement metric, not just its variance.

Photoreceptor, amacrine and C2/C3 transmitter identity is not actually unknown. The signs are
now fixed from published neurotransmitter assignments, **checked from primary sources in this
pass [V-live]**, and pre-registered in `PREREGISTRATION.md` §5 together with their evidence
grades. The short version:

| Type | Published transmitter | Sign basis | Sign | Evidence |
|---|---|---|---|---|
| `C2` | GABA | inhibitory | **−1** | strong |
| `C3` | GABA | inhibitory | **−1** | strong |
| `R8` | histamine (+ acetylcholine co-transmission) | histamine gates a chloride channel → inhibitory at the columnar targets in this model | **−1** | strong for histamine; **the co-transmission is a real, documented violation of the one-transmitter premise** |
| `Am` | glutamate (candidate) | glutamate acts through the GluClα chloride channel in this circuit → inhibitory | **−1** | weakest of the four |

**The honest part.** Xiao et al. (2023, *Nature*) show R8 co-transmits histamine and
acetylcholine, segregated by postsynaptic partner: histamine to unicolumnar retinotopic
neurons, acetylcholine to multicolumnar accessory-medulla neurons. So R8 **genuinely violates
Dale's law in the single-transmitter sense**, and a one-sign-per-type constraint is known to
be wrong for it. We apply −1 because the multicolumnar cholinergic pathway targets neurons
that are not in this 65-type model — but this must be stated in any write-up, and it is a
large part of why arm C (Dale on unambiguous types only) is a funded, pre-registered arm
rather than a fallback.

Two required implementation checks before Stage 2, both cheap:

- **Identity check.** Confirm that flyvis's `Am` is the lamina amacrine cell. If it is a
  different neuron, the assignment above is wrong.
- **Override audit.** Print every (pre, post) pair whose sign the explicit table overrides,
  the synapse-count weight affected, and the modal sign that type already carries in the
  shipped connectome. `DaleSignResolver` already produces this report; commit it to the repo
  alongside the pre-registration.

---

## 6. Metrics and statistics

### 6.1 Three mechanism metrics — and one that is not a mechanism metric

**M2 — mechanism fingerprint (PRIMARY).** The **current decomposition**: presynaptic-type
current contributions into **T4c** at preferred (90°) and null (270°) direction,
`intensity=1`, via `MovingEdgeCurrentView.get_temporal_contributions()`
(**[V-r1]** present at `92b3845` in `flyvis/analysis/moving_edge_currents.py`). For seed *s*,
build the time-integrated signed current vector `m_s[k, d]` over K presynaptic cell types and
`d ∈ {PD, ND}`, then **L1-normalize the whole vector** (`m̂ = m / ‖m‖₁`) — essential, or the
statistic measures output gain rather than mechanism.

**[V-r1] science-6, and it matters:** `src/stability.py` currently implements a *different*
M2 — L1-normalized moving-edge **angular tuning curves** of the eight T4/T5 subtypes — and
calls it "the actual mechanism." **It is not.** Tuning curves are the model's task-relevant
*output*. Two networks with different circuits and identical T4c tuning are exactly the
degeneracy this project exists to detect, and an output fingerprint cannot separate them;
conversely, "less mechanism variance at equal task error" built on an output fingerprint is
partly circular. **The tuning-curve fingerprint is demoted to a clearly labelled SECONDARY
RESPONSE metric (M2r).** Mechanism and response are different claims and must not share a
name. M2-as-currents is also invariant to the presynaptic gain/strength trade-off that the
tuning-curve version is not.

**M1 — circular dispersion of preferred direction (secondary).** Per seed, θ from
`preferred_direction(ensemble.moving_edge_responses())` — T4 subtypes at `intensity=1`, T5 at
`intensity=0`. Circular variance `1 − |mean(exp(iθ))|`; per-seed scalar = absolute angular
deviation from the arm's circular median. **T4c is primary**; Holm-correct if all eight
subtypes are reported.

**M3 — biological-agreement dispersion (secondary).** Per seed, the four correlations the
reproducibility paper plotted: FRI-corr, T4-tuning-corr, T5-tuning-corr, DSI-corr. Fisher-z
transform, then dispersion across seeds. **Report the mean shift and the dispersion
separately** — they are different claims, and the Dale sign policy can move the mean (§5.1).

### 6.2 The test: why PERMDISP is primary and Brown–Forsythe is not

**Do not use the F-test for variance ratios.** Its Type-I error is driven by kurtosis, not
variance, and does not improve with n. **[V-live]** — measured in this pass by running
`src/stability.py`, nominal α=0.05, n=50/arm:

| True distribution | F-test FPR | Brown–Forsythe FPR |
|---|---|---|
| Normal | 0.047 | 0.041 |
| t₅ (heavy-tailed) | **0.180** | 0.047 |
| Lognormal (skewed) | **0.537** | 0.051 |

(v1 quoted **0.329** for the lognormal F-test. That was wrong in the direction that
*understates* the failure; the module's own smoke test prints 0.537 and the round-1 reviewer
measured 0.564 with the same generator.)

**But Brown–Forsythe is not calibrated for the statistic this project actually consumes**
(science-3, **[V-r1]**). `deviations_from_center(center="medoid")` returns one exact zero per
arm — the medoid's own deviation. In high dimension that zero is a massive leverage point:
with 96-dimensional fingerprints it contributes **84% of Levene's denominator**, BF's size
collapses to ~0.000, and its power at a **true 3× dispersion ratio is 0.007**. Running the
module's own synthetic generator at N=48/arm, BF's null rejection rate is 0.013, not 0.05.
The obvious alternative is worse: a rank test on the same deviations has FPR 0.177, because
the deviations are not independent.

**So the verified power table below is correct for the i.i.d. normal scalar it simulates and
irrelevant to the endpoint we will report.** **[V-live]** (reproduced here by running the
module; the round-1 reviewer independently re-derived it with 20,000 reps):

| Variance ratio | 1.5× | 1.75× | 2× | 2.5× | 3× | 4× |
|---|---|---|---|---|---|---|
| N/arm, normal | 225 | 120 | **80** | 49 | 36 | 24 |
| N/arm, t₅ | 330 | 173 | 114 | 67 | **48** | 32 |

**Locked (D3):**

- **PRIMARY: PERMDISP** (Anderson 2006, *Biometrics* 62:245–253) on the M2 distance matrix.
  Its label permutation handles the dependence between deviations correctly, which is the
  specific thing BF-on-medoid gets wrong.
- **SECONDARY: Brown–Forsythe** on distance-to-centre, reported with its miscalibration
  stated, and computed with `center="mean_pairwise"` (no exact zero) as well as `"medoid"` so
  the difference is visible.
- **Required N is re-derived by simulating the actual pipeline end to end** — synthetic
  ensembles with a known dispersion ratio → `mechanism_distance_matrix` → PERMDISP → size and
  power. That simulation and its output are in `PREREGISTRATION.md` §6. **Do not quote the
  normal-scalar table for a distance-based endpoint.**
- **`permdisp()` is a from-scratch implementation that has never been checked against
  `vegan::betadisper` or `skbio`** (its own docstring says so, **[V-live]** confirmed by
  reading it). Anderson's imaginary-axis correction is implemented as `sqrt(|real − imag|)`
  and implementations differ on the absolute value. **Validating it on one dataset against
  one of those is a blocking prerequisite for Stage 2**, now that it is the primary test.

**Effect size, always reported alongside p:** `log(s²_A / s²_B)` with a BCa bootstrap CI,
resampling seeds within arm. It is reported for every primary result and **it does not
decide** — `PREREGISTRATION.md` §3 names the one statistic that does.

**Task-error equivalence:** TOST, margin **0.08 validation-loss units** (§6.3).

### 6.3 The equivalence margin (D8 / science-8)

v1 suggested ±5% of the baseline median. **[V-live]** At the released ensemble's mean of
5.3143 that is **0.2657 loss units = 3.53 across-seed SDs**, and at any N we can afford it
would pass automatically — a Monte-Carlo TOST at that margin reaches 80% power at **N = 4 per
arm**. Declaring "no cost to task error" that way is meaningless.

**Locked margin: 0.08 validation-loss units** — Karaneen et al.'s own verbatim figure
("cluster mean errors differed by less than 0.08"), i.e. the difference their paper says is
already enough to reorder which cluster matches biology. Relative to the released ensemble
(SD 0.0752) that is **1.06 across-seed SDs**.

**[V-live] N required for TOST at margin 0.08**, stated separately from the dispersion N, by
Monte-Carlo (two-sample pooled-variance TOST, α=0.05 each side, 80% power, SD=0.0752):

| True difference between arms | in SDs | **N per arm** |
|---|---|---|
| 0.00 | 0.00 | **16** |
| 0.01 | 0.13 | 17 |
| 0.02 | 0.27 | 21 |
| 0.03 | 0.40 | 29 |
| 0.04 | 0.53 | 45 |
| 0.05 | 0.66 | 79 |
| 0.06 | 0.80 | 176 |

**Reading:** the honest margin is affordable. If the arms truly perform identically, TOST
needs ~16 seeds/arm — far below the dispersion N. The risk is not the margin, it is a real
arm difference: at a true gap of 0.04 loss units TOST already needs 45/arm, and at 0.06 it
needs 176/arm, which we cannot afford. **Pre-registered consequence:** if TOST fails at the
achieved N, report it as "equivalence not established at margin 0.08 with N=…", give the
observed difference and its CI, and **do not widen the margin**.

### 6.4 Mandatory diagnostics: the optimizer-geometry confound (science-2)

Round 1's second-most-serious finding, and the reason the following are pre-registered as
*mandatory reported diagnostics* rather than optional extras.

flyvis uses Adam at lr 5e-5 (**[V-r1]**, `config/scheduler/scheduler.yaml`). Adam's update is
gradient-scale-invariant, so a *raw* parameter step is ~lr in both arms, but the *semantic*
step through softplus is `lr·(1 − exp(−m))`. At flyvis's own init (`m = 0.01/⟨N⟩`, median
5.0e-3 across the 605 type-pairs), a softplus arm moves its synaptic magnitudes **~201×
more slowly than a clamp arm at the median pair and up to 14,410× at the weakest**, and its τ
**20.9× more slowly** **[V-r1]**. Both arms start from the *same* deterministic init (only
`nodes_bias` is seeded). Less movement from a shared start produces less across-seed
dispersion by construction.

Putting softplus in both arms (D1) removes this as an *arm* confound. It does **not** remove
it as a *result* confound, because the CfC gate and the Dale sign rule change the geometry
too. So, per arm, always report:

- **(a) M2 dispersion at step 0 (untrained).** If arm B is already lower before training,
  the headline is an initialization artifact and the result is void.
- **(b) `‖θ_final − θ_init‖` in SEMANTIC space**, separately for `syn_strength` and τ.
- **(c) Mechanism distance from each seed to its own initialization.**
- **(d) Boundary-touch counts**: how many magnitudes sit within ε of zero at the end.
- **(e) The empirical distribution of learned τ relative to `dt`** (§1.4), per arm.
- **(f) M2 dispersion restricted to the 186 type-pairs that are never dead in the released
  ensemble** (604 − 418), as a clamp-free sensitivity analysis.

If (a) or (c) shows the arms were never free to move comparably, say so in the abstract.

### 6.5 Ground truth and two traps

All hard-coded in `flyvis/utils/groundtruth_utils.py` — a literature compilation, not a
download. The round-1 reviewer confirmed `stability.py`'s transcription is **bit-identical**
to flyvis for `tuning_curves`, `preferred_directions`, `motion_tuning` and `no_motion_tuning`
**[V-r1]**. FRI target: a binary ±1 polarity vector over **65 entries, 32 of them non-zero**
(`stability.py`'s docstring says 66 — wrong, fix it). DSI target: binary 0/1 over 18 cell
types. Tuning curves: 8 arrays × 12 angles, digitized from Maisak et al. 2013, *Nature*
500:212–216, Fig. 3g,h. Preferred directions: T4a/T5a=180°, T4b/T5b=0°, T4c/T5c=90°,
T4d/T5d=270°.

- **flyvis's DSI is a vector-sum DSI, not (PD−ND)/(PD+ND).** Computed from the Maisak curves
  the flyvis-style values are ~0.25–0.60 where the classic index gives ~0.48–0.92 — a factor
  of **1.2–2×** (v1 said 1.5–2×; T4c 1.25, T4d 1.22 and T5d 1.48 are all below 1.5
  **[V-r1]**). The denominator takes `max` over the *intensity* dimension, so ON and OFF DSI
  for the same cell share the larger total — a reimplementation that normalizes per intensity
  will not reproduce flyvis numbers.
- **The degree convention for preferred direction is not agreed across the literature.**
  flyvis encodes T4a=180°; Henning et al. 2022 state "front-to-back or 0° tuning for layer
  A". The subtype→cardinal-direction mapping is agreed; the degree labelling is
  convention-dependent. **Score against flyvis's own `preferred_directions` dict.** (Drop
  v1's "clusters around six directions" paraphrase of Henning et al. — the paper's six refers
  to six optic-flow patterns, not six tuning directions.)

---

## 7. Kill criteria

Pre-registered. Check in order; do not renegotiate after seeing a result.

1. **The instability does not reproduce.** If Stage-0 mechanism dispersion across the
   released 50 models is not large relative to task-error dispersion (mechanism CV < 5 ×
   task-error CV), the premise is gone. **Stop.** Write up the negative reproduction attempt.
2. **Dale / CfC badly hurts task error.** If the Stage-1 gate fails (arm B median validation
   loss >10% above arm A on a matched run), **loosen** rather than abandon: (a) run arm C as
   the treatment (Dale on unambiguous types only); (b) reduce the CfC gate from per-neuron to
   per-cell-type (65 values), which also recovers `AutoDeref` amortization; (c) revert any
   `syn_count` retie. If it still fails, the constrained model class is not viable on this
   task — report that; it is a real finding about the cost of biological constraints.
3. **The constrained arm is equally unstable.** A null dispersion result at the
   pre-registered N is a **negative result, not a failure**. Reframe as *"structure is not
   enough"*: connectome topology plus Dale plus liquid dynamics still do not pin down a
   mechanism. This strengthens and generalizes the reproducibility paper's conclusion.
   Publish with the power analysis attached so the null is interpretable.
4. **Underpowered by budget.** If `CU_PER_RUN` cannot be cut enough to reach **N ≥ 36/arm**
   with the full §4.3 inventory, **do not run Stage 2 anyway.** Re-scope per §4.5. An
   underpowered dispersion comparison is worth less than nothing here, because criticizing
   exactly that is part of the stated contribution.
5. **Nondeterminism dominates.** If the determinism-floor gate fails (`PREREGISTRATION.md`
   §7 — a one-sided upper confidence limit on floor/signal, not a point estimate), we are
   measuring CUDA nondeterminism. Re-scope around deterministic training
   (`torch.use_deterministic_algorithms`, a deterministic scatter) first — and note that this
   would itself be a useful finding, since the reproducibility paper explicitly asks for
   "elimination of nondeterministic training elements."

---

## 8. Known unknowns

Listed so nobody re-derives them as if they were settled.

**Cost and compute**

- **Wall-clock per training run** at the flyvis configuration is **[UNKNOWN]** — not in the
  *Nature* paper, README, docs, CHANGELOG or source **[V-r1]**. This is why Stage 0 exists,
  and it is the single input that decides whether the design in §4.5 fits.
- **Peak GPU memory** at the default config (extent=15, batch 4, 19-frame BPTT over
  1,513,231 edges) is **[UNKNOWN]** and genuinely undocumented.
- **Colab CU/hr rates** are third-party; Google publishes none **[V-r1]**. Treat ±20%. The
  in-session calibration overrides them.
- **Per-grant CU expiry dates** are not exposed anywhere.

**flyvis internals**

- **How the released 50 models differ from one another** — identical recorded configs,
  `seed: 0` everywhere. Unresolved whether the diversity came from unrecorded launch
  overrides or from training nondeterminism **[V-r1]**. **Set your own seeds.**
- Whether the maintainers intend `NetworkDynamics` to be subclassable as an `nn.Module`. It
  works **[V-r1]** but is undocumented with no upstream test coverage.
- Whether the released checkpoints are bitwise the weights behind the *Nature* figures.
  Parameter count (734) and ensemble size (50) match; no statement guarantees it.
- **The 605 / 604 and 2,140 / 2,355 gaps.** The shipped `fib25-fib19_v2.2.json` has 605
  distinct `(source_type, target_type)` pairs but the checkpoints carry 604
  `edges_syn_strength` values, and 2,140 `(pair, Δu, Δv)` triples versus 2,355
  `edges_syn_count` values **[V-r1]**. Presumably extent / `n_syn_fill` effects, but nothing
  in the plan or the modules explains them, while "604" is quoted everywhere as if it came
  from the JSON. **Explain both gaps in Stage 0 or stop quoting the provenance.**
- `ensemble_total_variance` uses population variance (`ddof=0`) while every other variance in
  `stability.py` uses `ddof=1` **[V-live]**. Harmless for a ratio at equal N, but it means the
  reported total variance is not on the same scale as the deviations feeding Brown–Forsythe.
  Say which is reported, in the pre-registration.
- **MPI-Sintel is 5,627,783,629 bytes (5.24 GiB)** **[V-r1]**, served over plain HTTP from an
  academic host, with no timeout anywhere in flyvis's fetch path and a bare
  `assert not sintel_zip.exists()` that makes a resumed download fail with no message. Any
  notebook that touches it needs a bounded, resumable pre-download and a partial-file
  cleanup. Synthetic stimuli (flashes, moving bars/edges) need no Sintel at all.
- Whether `flyvis download-pretrained` keeps working: it depends on a hardcoded Drive folder
  id and API key **[V-r1]** — note the *folder* id is hardcoded; the two *file* ids are
  listed at runtime, so any file id written in our code is **[ASSUMED]**, not verified.
  There is no Zenodo or HuggingFace mirror. **Archive the 3.4 MB ourselves.**

**The papers**

- Numeric values in Karaneen et al. Fig. 1 and its supplement were not extracted (no text
  layer). We have the caption text and the "<0.08" figure.
- How Karaneen et al. seeded their two new ensembles. Their published fork contains six files
  differing from upstream, three of them plotting scripts **[V-r1]** — no training scripts,
  no configs, no result data.
- The Lappalainen et al. *Nature* Methods text was never read directly (nature.com redirects
  to auth). Every metric definition here comes from flyvis source, which is stronger for
  reimplementation but means we cannot quote the paper's prose.

**Literature numbers in the code that read as authoritative but are not [V-r1]**

- `LITERATURE_DSI_CLASSIC` in `stability.py`: the J Neurosci ranges 0.75–0.85 (GCaMP6f) and
  0.35–0.45 (ArcLight) and their sample sizes (n=12 ROIs/N=8 flies; n=22/N=9). The paper is
  real and correctly identified (J Neurosci 43(14):2497, PMID 36849417, PMC10082464); the
  numbers are **[UNKNOWN]**.
- The Fisher 2015 "DSI > 0.5 inclusion threshold" and the Gruntman & Reiser
  `(PDmax − NDmax)/PDmax` formula: **[UNKNOWN]**.

**Design**

- The power tables assume a single pre-registered scalar, i.i.d. seeds, and equal means
  between arms. Testing all eight T4/T5 subtypes with Holm correction raises required N by
  roughly 25–40% **[ASSUMED]**; that case was not simulated.

---

## 9. Sources

**Premise and comparison papers**

- Karaneen, Schomburg & Chklovskii (2026). *Reproducibility and model-selection stability in
  connectome-constrained circuit modeling.* bioRxiv `10.64898/2026.04.18.717873`.
  Analysis code: [eschombu/flyvis @ matters-arising](https://github.com/eschombu/flyvis/tree/matters-arising)
- [*Topological Sensitivity in Connectome-Constrained Neural Networks*, arXiv 2604.04033](https://arxiv.org/abs/2604.04033)
- Lappalainen et al. (2024). *Connectome-constrained networks predict neural activity across
  the fly visual system.* Nature 634:1132–1140. [PMC11525180](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- [TuragaLab/flyvis](https://github.com/TuragaLab/flyvis) @ `v1.2.0` = `92b3845` ·
  [docs](https://turagalab.github.io/flyvis)

**Neurotransmitter assignments for the Dale sign table (§5.1)** — all read from primary
source in this pass **[V-live]**

- Kolodziejczyk A, Sun X, Meinertzhagen IA, Nässel DR (2008). *Glutamate, GABA and
  acetylcholine signaling components in the lamina of the Drosophila visual system.*
  PLoS ONE 3(5):e2110. doi:10.1371/journal.pone.0002110. — "GABA is the likely transmitter of
  centrifugal neurons C2 and C3"; "Glutamate may thus be used as a transmitter in amacrine
  neurons for wide-field interconnections."
- Xiao N, Xu S, Li Z-K, et al. (2023). *A single photoreceptor splits perception and
  entrainment by cotransmission.* Nature 623(7987):562–570. doi:10.1038/s41586-023-06681-6.
  — R8 co-transmits histamine and acetylcholine; histamine → unicolumnar retinotopic neurons,
  acetylcholine → multicolumnar accessory-medulla neurons.
- Davis FP, Nern A, Picard S, Reiser MB, Rubin GM, Eddy SR, Henry GL (2020). *A genetic,
  genomic, and computational resource for exploring neural circuit function.* eLife 9:e50901.
  doi:10.7554/eLife.50901. — "the main GABAergic inputs to L1 and L2 are C2 and C3 neurons";
  R8 expresses both cholinergic and histaminergic markers.
- Hardie RC (1989). *A histamine-activated chloride channel involved in neurotransmission at a
  photoreceptor synapse.* Nature 339:704–706. PMID 2472552. — histamine is inhibitory at the
  photoreceptor synapse.
- Molina-Obando S, Vargas-Fique JF, Henning M, Gür B, Schladt TM, Akhtar J, Berger TK,
  Silies M (2019). *ON selectivity in the Drosophila visual system is a multisynaptic process
  involving both glutamatergic and GABAergic inhibition.* eLife 8:e49373.
  doi:10.7554/eLife.49373. — "all ON responses are lost upon pharmacological block or genetic
  loss of GluClα".
- Raghu SV, Borst A (2011). *Candidate glutamatergic neurons in the visual system of
  Drosophila.* PLoS ONE 6(5):e19472. PMID 21573163. — amacrine among the candidate
  glutamatergic medulla-intrinsic types.

**Statistics and ground truth**

- Anderson MJ (2006). *Distance-based tests for homogeneity of multivariate dispersions.*
  Biometrics 62:245–253. (PERMDISP)
- Brown MB & Forsythe AB (1974). JASA 69:364–367.
- Maisak et al. (2013). *A directional tuning map of Drosophila elementary motion detectors.*
  Nature 500:212–216. [PMID 23925246](https://pubmed.ncbi.nlm.nih.gov/23925246/)
- Henning, Ramos-Traslosheros, Gür & Silies (2022). Sci Adv 8:eabi7112.
  [PMC8769539](https://pmc.ncbi.nlm.nih.gov/articles/PMC8769539/)

**Stage 4 (optional)**

- *Sexual dimorphism in the complete Drosophila male central nervous system connectome.*
  Cell (2026), `S0092-8674(26)00942-6`; preprint bioRxiv `10.1101/2025.10.09.680999`
  ([PMC12636603](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/)). **Cite the version you
  use — the counts differ (§5, Stage 4).**

**Internal**

- [`PREREGISTRATION.md`](PREREGISTRATION.md) — the committed analysis tuple and decision rule.
- `CRITIQUE_ROUND1.md` — the round-1 adversarial review, with the evidence behind every
  **[V-r1]** label in this file.
