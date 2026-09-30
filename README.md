# flyvisu

Are connectome-constrained models of the fruit fly's visual system **mechanistically
identifiable**? When the same model is retrained from a different random seed, does it learn the
same circuit, or only reach the same loss?

A 2026 reproducibility study (Karaneen, Schomburg & Chklovskii, bioRxiv
`10.64898/2026.04.18.717873`) retrained the flyvis model of the *Drosophila* optic lobe
(Lappalainen et al., *Nature* 2024) fifty times and found **stable task error but unstable
circuit mechanisms**. This repository tests whether biological constraints the connectome does
not encode, **Dale's law** (one sign per presynaptic cell type) and **state-dependent
("liquid", CfC) time constants**, make different seeds converge on one mechanism without costing
task performance.

## Status, 1 October 2026

Stage 0 (feasibility, premise, cost) is complete. Stage 2 (the two-arm seed comparison) has not
started: the pre-registered design does not yet fit the compute budget.

| Question | Answer | Evidence |
|---|---|---|
| Does the premise reproduce? | Yes. Across the 50 released flyvis models, minimum validation loss varies by **1.42 %** (CV). | `FINDINGS.md`, Stage 0 |
| Is the mechanism unstable across seeds? | Yes. The pre-registered fingerprint (signed currents into the central T4c cell during a moving edge) has **CV 0.59** (95 % CI 0.50 to 0.67), **41×** the loss CV. Go criterion 1 needs 5×. In 19 of 50 models Mi9 supplies under 1 % of T4c's input, in 11 Mi4 does. | `bench/results/m2_released_local_2026-09-19.json` |
| Do both arms run inside flyvis? | Yes. `verify_seam` passes 22 of 22 tensor-level checks on CPU and CUDA. A 58-agent code audit confirmed 27 issues (6 critical, among them "the two arms were the same model"); the code-level ones are fixed, and the design-level gaps it raised are listed in `FINDINGS.md`. | `src/flyvis_seam.py` |
| What does a full 250k-step run cost? | About **15 compute units** on a Colab L4 running four seeds at once (28 on a single T4). A local RTX A2000 laptop GPU trains at T4 speed (0.36 s/step) for free. | `bench/`, `FINDINGS.md` |
| How many seeds does the primary test need? | **30 per arm** for 80 % power at a 2× dispersion ratio, simulated on the measured fingerprint geometry (24 with the pre-registered synthetic generator, 12 with a Gaussian fit). | `bench/results/m2_power_2026-09-19.json` |
| Can runs be stopped early? | No. Over 60,000 steps (a quarter of flyvis's schedule) both arms' validation loss sits at 1207 ± 1 from step 3,000 on, and the fingerprint, measured at all 62 checkpoints, moves 0.03 and stays 0.39 away from every trained model. | `runs/long60k_2026-09-19/`, `bench/results/long60k_2026-09-19/` |

**Next step:** one unmodified flyvis run to 250,000 steps with checkpoints, to establish that the
flyvis 1.2.0 training recipe reproduces the released models' loss and mechanism at all, and where
in the schedule the mechanism forms. The open decisions (budget, thresholds to restate on the
measured loss scale, a time-resolved fingerprint) are listed at the end of `FINDINGS.md`.

## Read in this order

1. [`FINDINGS.md`](FINDINGS.md): the running log of everything measured, fixed and decided, with
   numbers as measured and the files they came from.
2. [`PREREGISTRATION.md`](PREREGISTRATION.md): the committed analysis (one primary metric, one
   primary test, one decision rule). Append-only.
3. [`PLAN.md`](PLAN.md): the reasoning, stage gates, kill criteria and budget arithmetic, as
   planned before Stage 0.
4. [`CRITIQUE_ROUND1.md`](CRITIQUE_ROUND1.md): the adversarial review that produced the current
   design.
5. [`docs/`](docs/): three illustrated, animated pages. `fly-circuit-anatomy.html` walks through
   the model in eleven plates, three of them built from our measurements;
   `fly-circuit-map.html` is the project map; `visu-anatomia.html` is the Portuguese version made
   for the VISU website. Open them in a browser, or enable GitHub Pages on `/docs`.

## What is new here, and what is not

**Already flyvis, not ours:** the recurrent network on the fixed connectome topology; cell-type
parameter tying (604 synaptic strengths per source and target type pair, 65 biases, 65 time
constants); the `sign × synapse count × strength` weight factorization; fixed synapse signs;
learnable per-type time constants; non-negative magnitudes.

**Shared substrate, identical in both arms:** a softplus magnitude parameterization in place of
flyvis's clamp at zero, and an exact exponential integrator in place of forward Euler. In the
released ensemble 418 of 604 strengths are exactly zero in some models and not in others, an
artifact of the clamp. Softplus removes that point mass by construction, so it sits in both arms;
otherwise a projection operator, not biology, would pre-explain any difference.

**The increment tested:** Dale's law (flyvis fixes signs per type *pair*; four presynaptic types,
R8, Am, C2 and C3, emit both signs in the shipped connectome, and Dale's law flips 10 of 604 pairs,
8,590 synapses), CfC time constants gated by each neuron's own state, and a pre-registered
mechanism-dispersion statistic with a properly powered test. Both arms train exactly 929
parameters.

**Why the released 50 models are a reference, not the control arm:** they were trained for
250,000 steps on other hardware and another flyvis build, all record `seed: 0`, and models 000 and
007 are byte-identical, so their spread is across something unrecorded rather than across seeds.
Both arms are therefore trained by us, under identical conditions, with recorded seeds.

## Repository layout

```
flyvisu/
├─ README.md                      this page
├─ FINDINGS.md                    the log of results and decisions
├─ PREREGISTRATION.md             the committed analysis
├─ PLAN.md                        plan, budget, gates, kill criteria
├─ CRITIQUE_ROUND1.md             round-1 adversarial review
├─ 00_stage0_setup_and_benchmark.ipynb
│                                 Colab notebook for Stage 0 (install, premise, feasibility, timing)
├─ build_stage0_notebook.py       generates the notebook; edit this, not the .ipynb
├─ src/
│  ├─ flyvis_seam.py              the two arms as flyvis classes; verify_seam (22 tensor checks)
│  ├─ dale_typed_cfc.py           standalone parameterization and its self-tests
│  └─ stability.py                M2 fingerprint, distances, PERMDISP, power simulation
├─ bench/
│  ├─ bench_gpu.py, bench_phases.py        step profile, concurrency, torch.compile
│  ├─ bench_curve.py                       learning curves, stock controls, LR sweep, long runs
│  │                                       with checkpoints mirrored to Google Drive
│  ├─ bench_eval_released.py               released models under flyvis 1.2.0's own loss
│  ├─ m2_released.py                       M2 on released models or on every checkpoint of a run
│  ├─ m2_power.py                          PERMDISP power on the measured geometry
│  ├─ m2_curve_analysis.py                 does the fingerprint settle before the loss?
│  ├─ tuning_released.py                   12-direction tuning of a released model's T4/T5 cells
│  └─ results/                             every result quoted in FINDINGS.md, as JSON
│     └─ explainer/                        data, templates and build scripts for docs/
├─ runs/long60k_2026-09-19/       the 60,000-step run: 62 checkpoints per arm, loss records
└─ docs/                          the illustrated pages (self-contained HTML)
```

## Reproducing

**Environment.** Python 3.12 (flyvis requires Python below 3.13), flyvis 1.2.0 installed from
`https://github.com/TuragaLab/flyvis` at tag `v1.2.0` (commit `92b3845`), PyTorch with CUDA for
training. On Windows, install `torch` and `torchvision` together from the CUDA wheel index;
installing flyvis alone pulls a CPU-only torch through torchvision. Set `FLYVIS_ROOT_DIR` before
importing flyvis; it decides where data and models live and is read at import time.

```bash
# self-tests, no GPU needed
python src/stability.py
python src/dale_typed_cfc.py
python -c "import sys; sys.path.insert(0, 'src'); import flyvis_seam as fs; fs.verify_seam(extent=15)"

# the 50 released models (flyvis's own download), then the fingerprint and its power
flyvis download-pretrained
mkdir -p out
python bench/m2_released.py compute out/m2.json          # roughly 30 s per model on a laptop CPU; see its docstring for sharding
python bench/m2_released.py analyze out/m2_summary.json out/m2.json
python bench/m2_power.py

# training on Colab with Google Drive mounted (checkpoints are mirrored to Drive), then M2 per
# checkpoint; to reuse our run, copy runs/long60k_2026-09-19/long_A to $FLYVIS_ROOT_DIR/results/flow/curve/long_A
python bench/bench_curve.py long 60000 <run_name>
python bench/m2_released.py checkpoints out/m2_long_A.json flow/curve/long_A
python bench/m2_curve_analysis.py out/m2_long_A.json out/m2_long_B.json

# the illustrated pages
python bench/results/explainer/build_anatomy.py
python bench/results/explainer/visu/build_visu.py
```

Stage 0 itself is `00_stage0_setup_and_benchmark.ipynb` on Colab, ordered cheapest cell first;
only the timing section spends compute units. Two flyvis details that cost time if missed: the
released models need `best_checkpoint_fn_kwargs={"validation_subdir": "validation",
"loss_file_name": "loss"}`, and flyvis's `datamate` dependency cannot rewrite `.h5` files on
Windows without the patch in `bench/m2_released.py` (`patch_datamate_for_windows`).

## Data, code and attribution

- **flyvis** (Lappalainen et al., *Nature* 2024) is used as a dependency and not vendored here;
  see its repository for its license. The 50 pretrained models are downloaded with flyvis's own
  command and are not redistributed.
- **MPI-Sintel** (Butler, Wulff, Stanley & Black, ECCV 2012) is derived from the open movie
  *Sintel*, © Blender Foundation, durian.blender.org, CC BY 3.0. This repository contains only a
  small derived excerpt used by the illustrated pages: 19 frames of the `alley_2` training sequence
  rendered to 721 hexagonal luminance values (`bench/results/explainer/sintel_clip.json`), a
  contact sheet and an animated GIF.
- `CRITIQUE_ROUND1.md` quotes flyvis source code; a Google API key hard-coded in that source has
  been redacted from the quotation.
- Documents written during the work refer to a private Google Drive folder, Colab sessions and
  private pages; those links will not open outside the team. The pages themselves are reproduced
  in `docs/`.

## License

No license has been chosen yet. Until one is added, no rights to reuse the code or documents are
granted.
