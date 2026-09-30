"""Mechanism variance across retraining seeds: measurement module.

This module is the measurement layer for a two-arm study of the form

    arm A (constrained)  = N independently seeded training runs
    arm B (baseline)     = N independently seeded training runs
    question             = does arm A have SMALLER mechanism variance than arm B?

It provides five things, in the order the study needs them:

    1. `mechanism_fingerprint`, `mechanism_distance`, `mechanism_distance_matrix`,
       `CurrentEnsemble` -- THE PRIMARY MECHANISM METRIC (M2): the
       presynaptic-type CURRENT DECOMPOSITION into the target cell at preferred
       and null direction, L1-normalized (section 3).
    2. `tuning_curve_response_distance`, `response_fingerprint`,
       `response_distance_matrix`, `TuningEnsemble` -- the SECONDARY RESPONSE
       metric (R1): moving-edge angular tuning curves (section 4). This is an
       OUTPUT metric. It is not a mechanism metric and must never be reported
       under that name.
    3. `across_seed_dispersion`, `ensemble_total_variance`, `circular_dispersion`,
       `permdisp` -- dispersion statistics for an ensemble of N models (section 5).
    4. `direction_selectivity_index`, `dsi_correspondence`,
       `preferred_direction_correspondence`, `dsi_correlation_to_known`
       -- correspondence to measured biology, against documented literature
       constants (sections 2 and 6).
    5. `permdisp` (PRIMARY TEST), `simulate_permdisp_power`,
       `n_per_arm_for_permdisp`, plus the secondary scalar machinery
       (`brown_forsythe_variance_test`, `permutation_variance_test`,
       `log_variance_ratio_ci`) -- inference for a VARIANCE (not a mean)
       difference between two arms, and the end-to-end power simulation that
       sizes the study (section 7).

Pure module: numpy / scipy / torch only. No I/O, no plotting, no flyvis import,
no notebook code. torch is optional and used only to accept torch tensors as
inputs (`_asnumpy`).

------------------------------------------------------------------------------
ROUND-2 CHANGES (locked design decisions D3 and D4) -- READ FIRST
------------------------------------------------------------------------------
D4 / science-6. MECHANISM != RESPONSE. The primary mechanism metric M2 is now
the CURRENT DECOMPOSITION (`mechanism_fingerprint` on a `CurrentEnsemble`):
time-integrated, signed, presynaptic-type current contributions into the target
cell type (pre-registered: T4c) at the preferred and null direction, jointly
L1-normalized. It is built from flyvis's
`MovingEdgeCurrentView.get_temporal_contributions`
(flyvis/analysis/moving_edge_currents.py; the API was re-read from the v1.2.0
source at commit 92b3845 on 2026-09-15 and the adapter below matches it).
The angular-tuning-curve fingerprint that previously carried the name
"mechanism" is now `response_fingerprint` / `tuning_curve_response_distance`
and is labelled a SECONDARY RESPONSE metric everywhere. Two networks with
different circuits and identical T4c tuning are exactly the degeneracy this
study exists to detect; a tuning-curve fingerprint cannot separate them, and a
"less mechanism variance at equal task error" claim built on an output
fingerprint is partly circular.

D3 / science-3. PERMDISP IS THE PRIMARY TEST. Brown-Forsythe on
medoid deviations is demoted to secondary and its miscalibration is documented
in `deviations_from_center` and `medoid_leverage`, and demonstrated by the
smoke test: the medoid contributes its own exact-zero distance, whose absolute
deviation from the median is then the single largest term in Levene's
denominator (measured in section [8] of the smoke test), which crushes the
statistic -- size well below nominal and near-zero power at a true 3x variance
ratio. The required N is re-derived by simulating THE ACTUAL PIPELINE end to
end (`simulate_permdisp_power`, `n_per_arm_for_permdisp`): synthetic
fingerprints -> L1 normalization -> Euclidean distance matrix -> PERMDISP.
The closed-form/scalar power table (`N_PER_ARM_REFERENCE_BF_NORMAL`,
`variance_test_power`, `n_per_arm_for_variance_ratio`) is retained ONLY as a
reference for an i.i.d. scalar endpoint and must not be quoted for the
distance-based endpoint this study reports.

THE PRE-REGISTERED PRIMARY ANALYSIS TUPLE IS `PRIMARY_ANALYSIS` BELOW. Every
other combination this module can compute is secondary by construction.

------------------------------------------------------------------------------
PROVENANCE, AND WHAT IS AN APPROXIMATION
------------------------------------------------------------------------------
Primary sources this module tracks:

  [Karaneen2026] Karaneen C, Schomburg EW, Chklovskii D (2026).
      "Reproducibility and model-selection stability in connectome-constrained
      circuit modeling." bioRxiv 2026.04.18.717873, posted 2026-04-20, CC-BY 4.0.
      Analysis code: github.com/eschombu/flyvis, branch `matters-arising` @ 8cc37c4.
  [Lappalainen2024] Lappalainen JK et al. (2024). "Connectome-constrained
      networks predict neural activity across the fly visual system."
      Nature 634:1132-1140. Software: github.com/TuragaLab/flyvis.
  [flyvis] TuragaLab/flyvis @ commit 92b3845 (tag v1.2.0, 2026-08-06). Metric
      definitions below are transcribed from
      `flyvis/analysis/moving_bar_responses.py`,
      `flyvis/analysis/flash_responses.py`,
      `flyvis/utils/groundtruth_utils.py`.
  [Maisak2013] Maisak MS et al. (2013). "A directional tuning map of Drosophila
      elementary motion detectors." Nature 500(7461):212-216.
      doi:10.1038/nature12320, PMID 23925246. -- source of the T4/T5 tuning
      curves (flyvis source comment: "from Maisak et al. 2013 Fig. 3 g, h").
  [Anderson2006] Anderson MJ (2006). "Distance-Based Tests for Homogeneity of
      Multivariate Dispersions." Biometrics 62:245-253. -- PERMDISP.
  [BrownForsythe1974] Brown MB, Forsythe AB (1974). "Robust Tests for the
      Equality of Variances." JASA 69(346):364-367.

HONEST SCOPE STATEMENT -- READ THIS BEFORE CITING THIS MODULE AS
"REIMPLEMENTING THE APRIL 2026 PAPER'S METRIC":

  [Karaneen2026] defines NO model-to-model distance, NO scalar instability
  statistic, and runs NO significance test. Its instability claim is
  qualitative: it clusters models (UMAP on T4c naturalistic-stimulus responses
  -> Gaussian mixture, BIC-selected) and shows that the map
  (cluster ranked by task error) -> (cluster matching biology) is not stable
  across retraining runs; and it plots violins of four per-model correlations
  to ground truth ("FRI", "T4 tuning", "T5 tuning", "DSI"), reporting that they
  are "weaker and more variable".

  Consequently:
    * Sections 3-5 (mechanism distance, response distance, across-seed
      dispersion) are an OPERATIONALIZATION, not a reproduction. The paper's
      clustering geometry (correlation distance on naturalistic-stimulus
      responses) is kept only in the SECONDARY response metric; the PRIMARY
      mechanism metric (section 3) is a current decomposition the paper does not
      compute at all. Both skip UMAP and the mixture model entirely: UMAP +
      BIC-selected GMM is not a metric, is not stable under reseeding, and
      cannot be plugged into a variance test; that instability is precisely the
      paper's point. See `mechanism_distance` and
      `tuning_curve_response_distance`.
    * Section 5's `dsi_correlation_to_known` and `fri_correlation_to_known` DO
      closely track the paper's four plotted quantities, because the paper
      delegates them to stock flyvis and flyvis's definitions were read from
      source. Deviations are flagged per function.
    * Section 2's DSI is a faithful transcription of the flyvis vector-sum
      formula INCLUDING its max-over-intensity normalization quirk, but it
      operates on peak responses supplied by the caller. flyvis's upstream
      `peak_responses` masking (speed-dependent column range and time window,
      `get_time_masks`) is NOT reimplemented here -- see
      `direction_selectivity_index`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as _ABCMapping
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline

try:  # optional: only used so callers can hand us torch tensors directly
    import torch as _torch
except Exception:  # pragma: no cover - torch is optional
    _torch = None

__all__ = [
    # ground truth constants
    "T4T5_CELL_TYPES",
    "KNOWN_TUNING_ANGLES_DEG",
    "KNOWN_TUNING_CURVES",
    "KNOWN_PREFERRED_DIRECTIONS_DEG",
    "KNOWN_DSI_VECTOR_SUM",
    "KNOWN_DSI_CLASSIC",
    "LITERATURE_DSI_CLASSIC",
    "MOTION_TUNING_TYPES",
    "NO_MOTION_TUNING_TYPES",
    "N_PER_ARM_REFERENCE_BF_NORMAL",
    "PRIMARY_ANALYSIS",
    "PIPELINE_POWER_TABLE_PERMDISP",
    # tuning-curve primitives
    "known_tuning_curves",
    "direction_selectivity_index",
    "direction_selectivity_index_classic",
    "preferred_direction",
    "angular_distance_deg",
    # PRIMARY mechanism metric (M2): presynaptic current decomposition
    "CURRENT_CONDITIONS",
    "PRIMARY_TARGET_TYPE",
    "CurrentEnsemble",
    "MechanismDistance",
    "time_integrated_currents",
    "mechanism_fingerprint",
    "mechanism_distance",
    "mechanism_distance_matrix",
    # SECONDARY response metric (R1): moving-edge angular tuning curves
    "TuningEnsemble",
    "ResponseDistance",
    "response_fingerprint",
    "tuning_curve_response_distance",
    "response_distance_matrix",
    # dispersion
    "fingerprint_distance_matrix",
    "ensemble_total_variance",
    "deviations_from_center",
    "medoid_leverage",
    "circular_dispersion",
    "circular_deviation_from_median",
    "EnsembleDispersion",
    "across_seed_dispersion",
    "PermdispResult",
    "permdisp",
    # correspondence to biology
    "DSICorrespondence",
    "dsi_correspondence",
    "dsi_correlation_to_known",
    "preferred_direction_correspondence",
    "flash_response_index",
    "fri_correlation_to_known",
    # two-arm variance inference
    "VarianceTestResult",
    "brown_forsythe_variance_test",
    "permutation_variance_test",
    "variance_f_test",
    "log_variance_ratio",
    "log_variance_ratio_ci",
    "PowerResult",
    "variance_test_power",
    "n_per_arm_for_variance_ratio",
    # end-to-end power simulation of the PRIMARY pipeline
    "PipelineSpec",
    "synthetic_current_ensemble",
    "calibrate_sigma_for_variance_ratio",
    "simulate_permdisp_power",
    "n_per_arm_for_permdisp",
]

_EPS_FLYVIS_DSI = 1e-15  # flyvis: dsi = vector_length / (normalization + 1e-15)
_EPS_FLYVIS_FRI = 1e-16  # flyvis: fri /= on_peak + off_peak + 1e-16


# =============================================================================
# 0. GROUND TRUTH CONSTANTS (with citations)
# =============================================================================

T4T5_CELL_TYPES: Tuple[str, ...] = (
    "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
)

#: Angles at which the literature tuning curves are tabulated, in degrees.
#: flyvis convention; `np.arange(0, 360, 30)` in
#: flyvis/analysis/moving_bar_responses.py::correlation_to_known_tuning_curves.
KNOWN_TUNING_ANGLES_DEG: np.ndarray = np.arange(0, 360, 30, dtype=float)

#: Measured T4/T5 angular tuning curves.
#:
#: SOURCE: [Maisak2013] Fig. 3g,h, digitized. Transcribed verbatim from
#: flyvis/utils/groundtruth_utils.py::tuning_curves (flyvis @ 92b3845), whose
#: source comment reads exactly "# from Maisak et al. 2013 Fig. 3 g, h".
#: Values are GCaMP calcium responses of T4/T5 axon terminals in the lobula
#: plate, normalized to roughly [0, 1], sampled every 30 deg.
#:
#: NOT VERIFIED: we did not see Maisak Fig. 3 itself, so the fidelity of the
#: digitization, its normalization, and which angle maps to index 0 are taken on
#: flyvis's authority. The recovered preferred directions (see
#: KNOWN_PREFERRED_DIRECTIONS_DEG) agree with the canonical 180/0/90/270 to
#: within one 30 deg bin for all eight subtypes, which corroborates the ordering.
KNOWN_TUNING_CURVES: Dict[str, np.ndarray] = {
    "T4a": np.array([0.04, 0.05, 0.165, 0.19, 0.29, 0.50, 1.0, 0.585, 0.31, 0.11, 0.06, 0.04]),
    "T4b": np.array([0.745, 0.785, 0.305, 0.265, 0.25, 0.335, 0.285, 0.23, 0.14, 0.11, 0.26, 0.775]),
    "T4c": np.array([0.04, 0.2, 0.57, 0.775, 0.715, 0.325, 0.22, 0.13, 0.13, 0.22, 0.155, 0.11]),
    "T4d": np.array([0.07, 0.07, 0.12, 0.17, 0.21, 0.14, 0.21, 0.29, 0.67, 0.655, 0.7, 0.510]),
    "T5a": np.array([0.20, 0.295, 0.265, 0.20, 0.25, 0.65, 1.0, 0.59, 0.29, 0.21, 0.20, 0.22]),
    "T5b": np.array([0.83, 0.855, 0.405, 0.16, 0.175, 0.24, 0.30, 0.25, 0.13, 0.2, 0.30, 0.655]),
    "T5c": np.array([0.20, 0.44, 0.69, 0.595, 0.745, 0.525, 0.34, 0.255, 0.34, 0.295, 0.26, 0.28]),
    "T5d": np.array([0.17, 0.21, 0.21, 0.20, 0.19, 0.15, 0.185, 0.42, 0.85, 0.79, 0.685, 0.565]),
}

#: Canonical preferred directions, degrees, FLYVIS CONVENTION.
#: SOURCE: flyvis/utils/groundtruth_utils.py::preferred_directions, which encodes
#: the four-cardinal-direction map of [Maisak2013] (T4/T5 subtypes a-d tile
#: lobula plate layers 1-4 = front-to-back, back-to-front, upward, downward).
#:
#: CONVENTION WARNING -- do not import a degree value from another paper without
#: checking its axis convention. Henning et al. 2022 (Sci Adv 8:eabi7112) state
#: verbatim "front-to-back or 0 degrees tuning for layer A ... back-to-front or
#: 180 degrees" -- i.e. a <-> 0 deg, the opposite labelling to flyvis's a <-> 180
#: deg. The subtype -> cardinal-direction mapping is agreed across sources; the
#: DEGREE labelling is convention-dependent (screen vs fly-centric coordinates and
#: the sign of the azimuth axis). Score models against THIS dict for internal
#: consistency with flyvis/[Lappalainen2024]/[Karaneen2026].
KNOWN_PREFERRED_DIRECTIONS_DEG: Dict[str, float] = {
    "T4a": 180.0, "T4b": 0.0, "T4c": 90.0, "T4d": 270.0,
    "T5a": 180.0, "T5b": 0.0, "T5c": 90.0, "T5d": 270.0,
}

#: Target DSI values under the FLYVIS VECTOR-SUM definition
#: (see `direction_selectivity_index`).
#:
#: PROVENANCE: DERIVED, NOT MEASURED. These are computed by applying the flyvis
#: vector-sum formula to the [Maisak2013] curves in KNOWN_TUNING_CURVES (single
#: intensity, single condition). They are NOT quoted from any paper. They are
#: verified at import-time-equivalent precision by the smoke test, which
#: recomputes them from KNOWN_TUNING_CURVES via
#: `direction_selectivity_index` and asserts agreement to 1e-3.
#:
#: Use these -- not literature DSI numbers -- when scoring a flyvis-style
#: vector-sum DSI. Mean over the eight subtypes is 0.397.
KNOWN_DSI_VECTOR_SUM: Dict[str, float] = {
    "T4a": 0.6026, "T4b": 0.3236, "T4c": 0.4473, "T4d": 0.4407,
    "T5a": 0.3358, "T5b": 0.3609, "T5c": 0.2548, "T5d": 0.4081,
}

#: Target DSI values under the CLASSIC definition (PD - ND) / (PD + ND).
#: PROVENANCE: DERIVED, NOT MEASURED -- same computation as above with
#: `direction_selectivity_index_classic` applied to KNOWN_TUNING_CURVES.
#: Note these run ~1.5-2x higher than the vector-sum values on the very same
#: curves. NEVER compare a flyvis vector-sum DSI to a literature classic DSI
#: without converting.
KNOWN_DSI_CLASSIC: Dict[str, float] = {
    "T4a": 0.9231, "T4b": 0.5468, "T4c": 0.5578, "T4d": 0.5385,
    "T5a": 0.6667, "T5b": 0.5475, "T5c": 0.4826, "T5d": 0.6038,
}

#: MEASURED classic-DSI values from the experimental literature, with the exact
#: formula each source used. There is NO single canonical population DSI for
#: T4/T5: the value depends on indicator, stimulus, speed and formula, so this is
#: a range-with-provenance table, not a point target. Treat DSI MAGNITUDE as a
#: weaker correspondence target than preferred direction or curve shape.
#:
#: Each entry: (low, high, formula, citation).
LITERATURE_DSI_CLASSIC: Dict[str, Tuple[float, float, str, str]] = {
    "T4_calcium_GCaMP6f_gratings": (
        0.75, 0.85, "(PDpeak - NDpeak) / (|PDpeak| + |NDpeak|)",
        "J Neurosci 2023, 'Voltage to Calcium Transformation Enhances Direction "
        "Selectivity in Drosophila T4 Neurons', PMC10082464; T4, slow gratings, "
        "n=12 ROIs / N=8 flies. DS decreases with increasing velocity.",
    ),
    "T4_voltage_ArcLight_gratings": (
        0.35, 0.45, "(PDpeak - NDpeak) / (|PDpeak| + |NDpeak|)",
        "J Neurosci 2023, PMC10082464; same cells, voltage indicator, "
        "n=22 ROIs / N=9 flies. Calcium ~0.8-0.9 vs voltage ~0.4 in the SAME "
        "cells -- the indicator, not the circuit, drives most of this gap.",
    ),
    "T4T5_inclusion_threshold": (
        0.5, 1.0, "not stated explicitly",
        "Fisher et al. 2015, Neuron, PMC4664581 -- uses DSI > 0.5 as the "
        "inclusion threshold for 'strongly direction tuned' T4/T5 ROIs.",
    ),
    "T5_whole_cell_bars": (
        float("nan"), float("nan"), "(PDmax - NDmax) / PDmax",
        "Gruntman & Reiser, eLife 2019 (elifesciences.org/articles/50706); T5 "
        "whole-cell, significantly > 0 for bar widths ~2.25/4.5/9 deg at "
        "14/28/56 deg/s. No population mean tabulated -- hence NaN.",
    ),
}

#: Cell types flyvis's `dsi_correlation_to_known` scores as direction selective
#: (binary target 1). SOURCE: flyvis/utils/groundtruth_utils.py::motion_tuning.
MOTION_TUNING_TYPES: Tuple[str, ...] = (
    "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
)

#: Cell types flyvis scores as NOT direction selective (binary target 0).
#: SOURCE: flyvis/utils/groundtruth_utils.py::no_motion_tuning.
NO_MOTION_TUNING_TYPES: Tuple[str, ...] = (
    "Tm1", "Tm2", "Tm4", "Tm9", "CT1(Lo1)", "Mi1", "Tm3", "Mi4", "Mi9", "CT1(M10)",
)

#: Reference N-per-arm table for Brown-Forsythe at alpha=0.05 (two-sided),
#: 80% power, equal n, normally distributed metric. These numbers come from an
#: independent simulation performed during the design research for this study;
#: `n_per_arm_for_variance_ratio` reproduces them APPROXIMATELY (it is itself a
#: Monte-Carlo search, so expect +-15% wobble at modest `n_sims`).
#: Heavy-tailed (t_5) metrics need roughly 1.4x these N.
N_PER_ARM_REFERENCE_BF_NORMAL: Dict[float, int] = {
    1.25: 600,   # ">600" in the reference table; treat as a floor, not a value
    1.5: 225,
    1.75: 120,
    2.0: 80,
    2.5: 49,
    3.0: 36,
    4.0: 24,
    5.0: 20,
}

#: THE PRE-REGISTERED PRIMARY ANALYSIS TUPLE (design decisions D3 + D4).
#:
#: This module can compute dozens of defensible analysis paths. Exactly ONE of
#: them is the study's endpoint; everything else is secondary and must be
#: labelled as such in any write-up. Fixing the tuple here is what makes the
#: endpoint real (see also the repository's PREREGISTRATION.md, which is the
#: authoritative copy -- if the two ever disagree, the committed file wins).
PRIMARY_ANALYSIS: Dict[str, Any] = {
    "metric_id": "M2",
    "metric": "presynaptic current decomposition into T4c at PD and ND",
    "builder": "mechanism_fingerprint(CurrentEnsemble.contributions)",
    "source": (
        "flyvis MovingEdgeCurrentView.get_temporal_contributions"
        "(contrast, angle, t_start, t_end, hide_source_types=None), "
        "time-integrated over [t_start, t_end], summed over cells, signed"
    ),
    "target_type": "T4c",
    "conditions": ("PD", "ND"),
    "source_type_set": "PRE-REGISTER EXPLICITLY (see CurrentEnsemble docstring)",
    "normalization": "l1 (joint over the whole source-type x condition vector)",
    "distance": "euclidean",
    "center": "centroid",
    "test": "PERMDISP (Anderson 2006), permutation of arm labels, n_perm=9999",
    "decision_rule": "PERMDISP p < 0.05, one analysis, no multiplicity budget spent",
    "effect_size": (
        "log(V_baseline / V_constrained) of ensemble_total_variance, "
        "BCa bootstrap CI over seeds -- REPORTED ALWAYS, never a decision rule"
    ),
    "secondary": (
        "R1 = tuning-curve response fingerprint (same pipeline); "
        "M1 = circular dispersion of T4c preferred direction; "
        "M3 = correspondence to biology (DSI / FRI); "
        "Brown-Forsythe on per-seed deviations (MISCALIBRATED, see D3); "
        "PERMDISP with center='median'",
    ),
}

#: SEEDS PER ARM for 80% power at alpha=0.05, PRIMARY endpoint, DERIVED BY
#: SIMULATING THE ACTUAL PIPELINE (design decision D3, replacing the scalar
#: table `N_PER_ARM_REFERENCE_BF_NORMAL` for this purpose).
#:
#: Pipeline simulated: synthetic current fingerprints (12 presynaptic types x
#: {PD, ND} = 24 dims) -> joint L1 normalization -> Euclidean distance matrix ->
#: PERMDISP(centroid) with 199 label permutations -> reject at p < 0.05.
#: Generated by `n_per_arm_for_permdisp` with n_sims=400 per grid point and
#: rng seed 20260915; the variance ratio is the REALIZED ratio of
#: `ensemble_total_variance` (calibrated by
#: `calibrate_sigma_for_variance_ratio`), not a nominal sigma ratio.
#: Reproduced within Monte-Carlo error by the smoke test for one cell.
#:
#: ROWS ARE INDEXED BY THE EFFECTIVE RANK p_eff OF THE ACROSS-SEED COVARIANCE,
#: AND THAT CHOICE MOVES THE ANSWER BY A FACTOR OF SEVERAL. p_eff = 1 means all
#: seeds differ along a single direction of the fingerprint (worst case for this
#: test, and the row to fund if you have no better information); p_eff = 24 is
#: isotropic variation. ESTIMATE p_eff FROM THE 50 RELEASED MODELS IN STAGE 0
#: -- it is free -- and pre-register the row before Stage 2.
#: `None` means the simulated grid (up to 300/arm) did not reach 80% power.
PIPELINE_POWER_TABLE_PERMDISP: Dict[str, Dict[float, Optional[int]]] = {
    "normal/p_eff=1": {1.5: None, 2.0: None, 2.5: 64, 3.0: 36, 4.0: 20},
}


# =============================================================================
# 1. ARRAY HELPERS
# =============================================================================

def _asnumpy(x: Any) -> np.ndarray:
    """Coerce numpy array / torch tensor / nested sequence to a float64 ndarray."""
    if _torch is not None:
        if isinstance(x, _torch.Tensor):
            x = x.detach().to("cpu").numpy()
        elif isinstance(x, (list, tuple)):
            # elements may be CUDA / requires-grad tensors, which np.asarray cannot convert
            x = [_asnumpy(v) for v in x]
    return np.asarray(x, dtype=np.float64)


def _check_1d(x: np.ndarray, name: str) -> np.ndarray:
    x = _asnumpy(x).ravel()
    if x.size < 2:
        raise ValueError(f"{name} must have at least 2 elements, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    return x


def _normalize_last_axis(x: np.ndarray, how: str) -> np.ndarray:
    """Normalize along the LAST axis. Shared by the mechanism and response paths.

    how="none": identity.
    how="l1":   x / sum|x|   -- strips overall gain, keeps the SHAPE (and, for a
                signed vector, the sign pattern and the relative balance of
                excitation and inhibition).
    how="l2":   x / ||x||_2.
    how="max":  x / max|x|.
    A zero vector is returned unchanged (denominator forced to 1).
    """
    if how == "none":
        return x
    if how == "l1":
        denom = np.abs(x).sum(axis=-1, keepdims=True)
    elif how == "l2":
        denom = np.sqrt((x ** 2).sum(axis=-1, keepdims=True))
    elif how == "max":
        denom = np.abs(x).max(axis=-1, keepdims=True)
    else:
        raise ValueError(f"unknown normalization {how!r}")
    return x / np.where(denom > 0, denom, 1.0)


def _match_names(names_a: Sequence[str], names_b: Sequence[str]) -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    """Intersect two name lists, preserving the order of the first.

    THIS IS THE "matched by name" STEP (cell types for the response metric,
    presynaptic source types for the mechanism metric). Two flyvis models share
    the connectome, so their name sets are normally identical and this is a
    no-op; the matching exists so that a model trained on a reduced or rewired
    connectome (e.g. a degree-preserving control arm) can still be compared on
    the names it has in common, rather than by positional index, which would
    silently compare T4c to T5d.
    """
    a = list(names_a)
    b = list(names_b)
    b_index = {ct: i for i, ct in enumerate(b)}
    shared = [ct for ct in a if ct in b_index]
    if not shared:
        raise ValueError("no shared names between the two models")
    ia = np.array([a.index(ct) for ct in shared], dtype=int)
    ib = np.array([b_index[ct] for ct in shared], dtype=int)
    return tuple(shared), ia, ib


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson r between two (..., n) arrays; 0.0 where a row is constant.

    Matches flyvis's `xr.corr(..., dim="angle").fillna(0.0)` convention: a
    degenerate (constant) tuning curve yields correlation 0, not NaN.
    """
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    na = np.sqrt((a * a).sum(axis=-1))
    nb = np.sqrt((b * b).sum(axis=-1))
    denom = na * nb
    num = (a * b).sum(axis=-1)
    out = np.zeros_like(num)
    good = denom > 0
    out[good] = num[good] / denom[good]
    return out


# =============================================================================
# 2. TUNING-CURVE PRIMITIVES (flyvis-tracking definitions)
# =============================================================================

def known_tuning_curves(
    cell_types: Sequence[str] = T4T5_CELL_TYPES,
    angles_deg: Optional[np.ndarray] = None,
    *,
    periodic: bool = False,
) -> np.ndarray:
    """Literature T4/T5 tuning curves, interpolated onto arbitrary probe angles.

    Definition
    ----------
    Let g_c be the 12-point [Maisak2013] curve for cell type c, tabulated at
    KNOWN_TUNING_ANGLES_DEG = {0, 30, ..., 330}. Returns

        G[c, k] = S_c(angles_deg[k])

    where S_c is the natural ("not-a-knot") cubic spline through
    (KNOWN_TUNING_ANGLES_DEG, g_c), extrapolated outside the tabulated range.

    Citation
    --------
    Curves: [Maisak2013] Fig. 3g,h via flyvis/utils/groundtruth_utils.py.
    Interpolation: flyvis/analysis/moving_bar_responses.py::get_known_tuning_curves
    uses `scipy.interpolate.interp1d(kind='cubic', fill_value='extrapolate')`,
    which is a not-a-knot cubic spline -- numerically the same object as
    `scipy.interpolate.CubicSpline(..., bc_type='not-a-knot')` used here.

    APPROXIMATION NOTE
    ------------------
    flyvis's interpolation is NOT periodic-aware: the spline treats 0 deg and
    330 deg as endpoints rather than neighbours, so values interpolated between
    330 and 360 deg are extrapolations and can be mildly wrong. We reproduce
    that behaviour by default so numbers match flyvis. Pass `periodic=True` to
    get the arguably-correct periodic spline instead; this will NOT match flyvis.

    Returns
    -------
    (len(cell_types), len(angles_deg)) array.
    """
    angles = KNOWN_TUNING_ANGLES_DEG if angles_deg is None else _asnumpy(angles_deg).ravel()
    out = np.empty((len(cell_types), angles.size), dtype=np.float64)
    for i, ct in enumerate(cell_types):
        if ct not in KNOWN_TUNING_CURVES:
            raise KeyError(
                f"no measured tuning curve for cell type {ct!r}; "
                f"available: {sorted(KNOWN_TUNING_CURVES)}"
            )
        y = KNOWN_TUNING_CURVES[ct]
        if periodic:
            x_ext = np.append(KNOWN_TUNING_ANGLES_DEG, 360.0)
            y_ext = np.append(y, y[0])
            spline = CubicSpline(x_ext, y_ext, bc_type="periodic")
            out[i] = spline(np.mod(angles, 360.0))
        else:
            spline = CubicSpline(KNOWN_TUNING_ANGLES_DEG, y, bc_type="not-a-knot",
                                 extrapolate=True)
            out[i] = spline(angles)
    return out


def direction_selectivity_index(
    peaks: np.ndarray,
    angles_deg: np.ndarray,
    *,
    average_conditions: bool = True,
    rectify: bool = True,
) -> np.ndarray:
    """Vector-sum direction selectivity index, flyvis definition.

    Definition (transcribed from flyvis @ 92b3845,
    flyvis/analysis/moving_bar_responses.py::direction_selectivity_index)
    ------------------------------------------------------------------
    For cell type c, intensity i, stimulus condition w (a (width, speed) pair)
    and probe angle theta:

        vector_length(c, i, w) = | sum_theta  peak(c, i, w, theta) * exp(1j*theta) |
        normalization(c, w)    = max_i  sum_theta | peak(c, i, w, theta) |
        DSI(c, i, w)           = vector_length(c, i, w) / (normalization(c, w) + 1e-15)
        DSI(c, i)              = mean_w DSI(c, i, w)                [average_conditions]

    THE NORMALIZATION QUIRK IS LOAD-BEARING AND INTENTIONAL. flyvis takes the
    MAX OVER THE INTENSITY AXIS in the denominator, so ON and OFF DSI for the
    same cell share the larger of the two total-response sums; the DSI for a
    cell's non-preferred contrast is therefore deliberately deflated. Any
    reimplementation that normalizes each intensity separately will NOT
    reproduce flyvis numbers (and therefore will not reproduce [Karaneen2026]'s
    "DSI" violin, which is stock flyvis).

    This is a VECTOR-SUM index, not (PD - ND) / (PD + ND). On the very same
    curves it runs 1.5-2x LOWER; see KNOWN_DSI_VECTOR_SUM vs KNOWN_DSI_CLASSIC
    and `direction_selectivity_index_classic`.

    Parameters
    ----------
    peaks : array, shape (..., n_intensities, n_conditions, n_angles)
        Peak responses. Leading axes (e.g. model, cell type) are broadcast over.
        Use `n_intensities=1` / `n_conditions=1` for a reduced experiment; with
        one intensity the max-over-intensity normalization degenerates to a
        plain sum, which is the correct limit.
    angles_deg : array, shape (n_angles,)
        Probe directions in degrees (flyvis default: 0, 30, ..., 330).
    average_conditions : bool
        If True (flyvis `average=True`), average over the condition axis and
        return shape (..., n_intensities). If False, keep it.
    rectify : bool
        Clip negative peaks to zero first. flyvis rectifies upstream inside
        `peak_responses` (`clip(min=0)` before `max` over frames), so True
        reproduces flyvis when handed raw (possibly negative) peaks.

    APPROXIMATION NOTE -- WHAT THIS DOES NOT DO
    -------------------------------------------
    flyvis computes `peaks` itself from simulated responses via
    `peak_responses`, which applies (a) a speed-dependent column/time mask
    (`get_time_masks`, `mask_between_seconds`) restricting the response window
    to when the moving edge is over the receptive field, then (b) `clip(min=0)`,
    then (c) `max` over frames. Steps (a)-(c) are NOT reimplemented here: this
    function consumes peaks the caller has already computed. If you want
    numbers comparable to published flyvis values, generate `peaks` with
    flyvis's own `peak_responses` (moving edge defaults: speeds
    (2.4, 4.8, 9.7, 13, 19, 25), offsets (-10, 11), intensities [0, 1],
    height 80, width 80, dt 1/200, t_pre = t_post = 1.0) and pass them in here.

    Returns
    -------
    (..., n_intensities) or (..., n_intensities, n_conditions) array.
    """
    peaks = _asnumpy(peaks)
    angles = _asnumpy(angles_deg).ravel()
    if peaks.ndim < 3:
        raise ValueError(
            "peaks must have shape (..., n_intensities, n_conditions, n_angles); "
            f"got ndim={peaks.ndim}. Add singleton axes for a reduced experiment."
        )
    if peaks.shape[-1] != angles.size:
        raise ValueError(
            f"last axis of peaks ({peaks.shape[-1]}) must match "
            f"len(angles_deg) ({angles.size})"
        )
    if rectify:
        peaks = np.clip(peaks, 0.0, None)

    phase = np.exp(1j * np.deg2rad(angles))
    vector_length = np.abs((peaks * phase).sum(axis=-1))          # (..., I, W)
    abs_sum = np.abs(peaks).sum(axis=-1)                          # (..., I, W)
    normalization = abs_sum.max(axis=-2, keepdims=True)           # (..., 1, W)
    dsi = vector_length / (normalization + _EPS_FLYVIS_DSI)       # (..., I, W)
    if average_conditions:
        dsi = dsi.mean(axis=-1)                                   # (..., I)
    return dsi


def direction_selectivity_index_classic(
    peaks: np.ndarray,
    angles_deg: np.ndarray,
    *,
    rectify: bool = True,
) -> np.ndarray:
    """Classic DSI = (PD - ND) / (PD + ND).

    Definition
    ----------
    With R(theta) the (rectified) peak response,

        theta_PD = argmax_theta R(theta)
        theta_ND = the probe angle closest to theta_PD + 180 deg
        DSI      = (R(theta_PD) - R(theta_ND)) / (R(theta_PD) + R(theta_ND) + 1e-15)

    Citation
    --------
    This is the form used by the experimental literature we compare against,
    e.g. J Neurosci 2023 PMC10082464: "DSI = (PDpeak - NDpeak) /
    (|PDpeak| + |NDpeak|)". Note Gruntman & Reiser (eLife 2019) and
    Henning et al. (Sci Adv 2022) instead use (PDmax - NDmax) / PDmax, which is
    a THIRD scale; this module does not implement that variant.

    NOT a flyvis function -- flyvis only ships the vector-sum index. Provided so
    model DSIs can be put on the same scale as measured values in
    LITERATURE_DSI_CLASSIC.

    Parameters
    ----------
    peaks : array, shape (..., n_angles)
    angles_deg : array, shape (n_angles,)

    Returns
    -------
    (...,) array.
    """
    peaks = _asnumpy(peaks)
    angles = _asnumpy(angles_deg).ravel()
    if peaks.shape[-1] != angles.size:
        raise ValueError("last axis of peaks must match len(angles_deg)")
    if rectify:
        peaks = np.clip(peaks, 0.0, None)

    pd_idx = np.argmax(peaks, axis=-1)
    # index of the probe angle nearest to (preferred + 180 deg), on the circle
    opposite = np.mod(angles[pd_idx] + 180.0, 360.0)
    diff = np.abs(np.mod(angles[None, :] - opposite.reshape(-1, 1) + 180.0, 360.0) - 180.0)
    nd_idx = np.argmin(diff, axis=-1).reshape(pd_idx.shape)

    r_pd = np.take_along_axis(peaks, pd_idx[..., None], axis=-1)[..., 0]
    r_nd = np.take_along_axis(peaks, nd_idx[..., None], axis=-1)[..., 0]
    return (r_pd - r_nd) / (np.abs(r_pd) + np.abs(r_nd) + _EPS_FLYVIS_DSI)


def preferred_direction(
    peaks: np.ndarray,
    angles_deg: np.ndarray,
    *,
    sum_axes: Tuple[int, ...] = (),
    rectify: bool = True,
    degrees: bool = False,
) -> np.ndarray:
    """Vector-sum preferred direction.

    Definition (flyvis/analysis/moving_bar_responses.py::preferred_direction,
    `average=True` branch)
    --------------------------------------------------------------------------
        theta_pref = angle( sum_{width, speed, theta} peak * exp(1j*theta) )

    i.e. the complex peak responses are summed over angle AND over the nuisance
    condition axes before taking the argument. Returned in radians on
    (-pi, pi] by default.

    Parameters
    ----------
    peaks : array, shape (..., n_angles); angle is the LAST axis.
    angles_deg : (n_angles,) probe directions in degrees.
    sum_axes : extra axes of `peaks` to sum the complex vector over before
        taking the angle (flyvis sums over width and speed). Given as axis
        indices into `peaks`; negative indices allowed, angle axis excluded.
    degrees : return degrees in [0, 360) instead of radians.
    """
    peaks = _asnumpy(peaks)
    angles = _asnumpy(angles_deg).ravel()
    if peaks.shape[-1] != angles.size:
        raise ValueError("last axis of peaks must match len(angles_deg)")
    if rectify:
        peaks = np.clip(peaks, 0.0, None)
    vec = (peaks * np.exp(1j * np.deg2rad(angles))).sum(axis=-1)
    if sum_axes:
        axes = tuple(ax if ax >= 0 else ax + peaks.ndim for ax in sum_axes)
        if (peaks.ndim - 1) in axes:
            raise ValueError("sum_axes must not include the angle axis")
        vec = vec.sum(axis=tuple(sorted(axes, reverse=True)))
    theta = np.angle(vec)
    return np.mod(np.rad2deg(theta), 360.0) if degrees else theta


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute circular distance between angles in degrees, in [0, 180].

    Definition: d(a, b) = | ((a - b + 180) mod 360) - 180 |.

    Tracks flyvis/analysis/moving_bar_responses.py::simple_angle_distance /
    angular_distance_to_known, which compare a model's preferred direction to
    KNOWN_PREFERRED_DIRECTIONS_DEG (T4 at intensity=1, T5 at intensity=0).
    """
    a = _asnumpy(a)
    b = _asnumpy(b)
    return np.abs(np.mod(a - b + 180.0, 360.0) - 180.0)


# =============================================================================
# 3. PRIMARY MECHANISM METRIC (M2): PRESYNAPTIC CURRENT DECOMPOSITION
# =============================================================================
#
# WHY THIS AND NOT TUNING CURVES (design decision D4, reviewer issue science-6)
# -----------------------------------------------------------------------------
# A tuning curve is what the network OUTPUTS. The question this study asks is
# whether independently seeded runs converge on the same INTERNAL CIRCUIT. Two
# networks that route completely different presynaptic currents into T4c and
# still produce the same T4c tuning curve are exactly the degeneracy the study
# exists to detect -- and a tuning-curve fingerprint cannot separate them. Worse,
# the headline claim is "less mechanism variance AT EQUAL TASK ERROR", so
# measuring "mechanism" with an output quantity makes the claim partly circular.
#
# The primary metric is therefore the decomposition of the input current into
# one target cell type (pre-registered: T4c) by PRESYNAPTIC CELL TYPE, at the
# preferred and the null direction, time-integrated and jointly L1-normalized.
# It answers "which presynaptic types supply the drive, in what proportion, and
# how does that allocation differ between PD and ND" -- a mechanism question.
#
# INVARIANCE PROPERTY, AND ITS LIMIT. Joint L1 normalization removes the overall
# current scale, so a model that doubles every synaptic weight and halves every
# gain is NOT counted as a different mechanism. It does NOT remove the
# presynaptic gain/strength trade-off within a type: a weaker synapse from a
# more active presynaptic type produces the same current as a stronger synapse
# from a quieter one, and M2 calls those the same mechanism. That is deliberate
# -- current is the causally relevant quantity for the postsynaptic cell -- but
# it must be stated, because it is the one substitution M2 is blind to.

CURRENT_CONDITIONS: Tuple[str, str] = ("PD", "ND")

#: Pre-registered target cell type for M2. T4c is the cell type both
#: [Lappalainen2024] and [Karaneen2026] cluster on, and the one flyvis's own
#: current-decomposition figure uses.
PRIMARY_TARGET_TYPE: str = "T4c"


@dataclass(frozen=True)
class MechanismDistance:
    """Result of a model-to-model MECHANISM distance (current decomposition)."""

    value: float
    per_source_type: Dict[str, float]
    metric: str
    source_types: Tuple[str, ...]
    target_type: str = PRIMARY_TARGET_TYPE


@dataclass(frozen=True)
class CurrentEnsemble:
    """Presynaptic current contributions for an ensemble of seeded models.

    Attributes
    ----------
    contributions : (n_models, n_source_types, n_conditions) array
        Time-integrated, SIGNED current from each presynaptic cell type into the
        target cell type, one entry per stimulus condition (default PD, ND).
        Sign convention: excitatory input positive, inhibitory negative, exactly
        as flyvis's current arrays carry it. Do NOT rectify.
    source_types : names indexing axis 1, in a FIXED, PRE-REGISTERED order.
    conditions : names indexing axis 2 (default ("PD", "ND")).
    target_type : the postsynaptic cell type the currents flow into.
    arm : optional label ("constrained" / "baseline").

    TWO TRAPS IN THE flyvis SOURCE -- BOTH ARE HANDLED BY
    `from_flyvis_current_view`, AND BOTH WILL SILENTLY CORRUPT THE STUDY IF YOU
    CALL flyvis DIRECTLY WITHOUT READING THIS
    ---------------------------------------------------------------------------
    (1) `MovingEdgeCurrentView.get_temporal_contributions` defaults to
        `hide_source_types="auto"`, which DROPS presynaptic types whose
        contribution falls below a data-dependent cut-off
        (`filter_cell_types_by_contribution`). The set of features would then
        differ from model to model and from arm to arm, which is not a
        comparison. Always pass `hide_source_types=None` and fix the source-type
        set yourself.
    (2) The `source_types` list it returns is produced by `sorting()`, i.e.
        ORDERED BY CONTRIBUTION MAGNITUDE, descending -- also data-dependent.
        Never zip it positionally across models; index by name.
    (Both read from flyvis/analysis/moving_edge_currents.py at commit 92b3845 on
    2026-09-15.)
    """

    contributions: np.ndarray
    source_types: Tuple[str, ...]
    conditions: Tuple[str, ...] = CURRENT_CONDITIONS
    target_type: str = PRIMARY_TARGET_TYPE
    arm: str = ""

    def __post_init__(self) -> None:
        c = _asnumpy(self.contributions)
        sts = tuple(self.source_types)
        conds = tuple(self.conditions)
        if c.ndim != 3:
            raise ValueError(
                f"contributions must be (n_models, n_source_types, n_conditions); got {c.shape}"
            )
        if c.shape[1] != len(sts):
            raise ValueError(
                f"contributions axis 1 = {c.shape[1]} but got {len(sts)} source types"
            )
        if c.shape[2] != len(conds):
            raise ValueError(
                f"contributions axis 2 = {c.shape[2]} but got {len(conds)} conditions"
            )
        if len(set(sts)) != len(sts):
            raise ValueError("source_types must be unique")
        if not np.all(np.isfinite(c)):
            raise ValueError("contributions contains non-finite values")
        object.__setattr__(self, "contributions", c)
        object.__setattr__(self, "source_types", sts)
        object.__setattr__(self, "conditions", conds)

    @property
    def n_models(self) -> int:
        return self.contributions.shape[0]

    @property
    def n_features(self) -> int:
        return self.contributions.shape[1] * self.contributions.shape[2]

    def model(self, i: int) -> np.ndarray:
        """(n_source_types, n_conditions) contributions of model i."""
        return self.contributions[i]

    @classmethod
    def from_flyvis_current_view(
        cls,
        current_view: Any,
        source_types: Sequence[str],
        *,
        contrast: float,
        angle: float,
        dt: float,
        t_start: float = 0.0,
        t_end: float = 1.0,
        reduce: str = "integral",
        arm: str = "",
        target_type: Optional[str] = None,
    ) -> "CurrentEnsemble":
        """Build the PRIMARY M2 ensemble from a flyvis `MovingEdgeCurrentView`.

        Verified against flyvis v1.2.0 (commit 92b3845,
        flyvis/analysis/moving_edge_currents.py), re-read from source on
        2026-09-15. The call made here is

            cv_pd, cv_nd, _returned_types, _colors = (
                current_view.get_temporal_contributions(
                    contrast=contrast, angle=angle,
                    t_start=t_start, t_end=t_end,
                    hide_source_types=None,      # trap (1): NEVER "auto"
                )
            )

        `get_temporal_contributions` internally does
        `at_contrast(contrast).at_angle(angle).between_seconds(t_start, t_end)
        .sum_over_cells()` for the PD view and the same at
        `(angle - 180) % 360` for the ND view, so the PD/ND pairing is flyvis's
        own and the currents are already summed over the target cell's input
        cells. `cv[source_type][:]` then yields that type's current trace, which
        is exactly how flyvis's own `get_summed_traces` reads them.

        The returned `_returned_types` is IGNORED on purpose (trap (2): it is
        sorted by contribution). Features are indexed by the `source_types` you
        pass, which is the pre-registered, fixed feature set; a missing type
        raises rather than silently shrinking the fingerprint.

        Parameters
        ----------
        current_view : a flyvis MovingEdgeCurrentView (duck-typed; this module
            never imports flyvis).
        source_types : the PRE-REGISTERED presynaptic types, in a fixed order.
        contrast, angle : flyvis stimulus selectors. `angle` is the PREFERRED
            direction to score; ND is taken as `(angle - 180) % 360` by flyvis.
        dt : simulation timestep in seconds (flyvis moving-edge default 1/200),
            used for the time integral.
        t_start, t_end : integration window in seconds (flyvis defaults 0, 1).
        reduce : "integral" (trapezoid, units A*s), "sum" (dt * sum) or "mean".

        Returns
        -------
        CurrentEnsemble with contributions of shape
        (n_models, len(source_types), 2).
        """
        out = current_view.get_temporal_contributions(
            contrast=contrast,
            angle=angle,
            t_start=t_start,
            t_end=t_end,
            hide_source_types=None,
        )
        cv_pd, cv_nd = out[0], out[1]
        contributions = time_integrated_currents(
            cv_pd, cv_nd, source_types, dt=dt, reduce=reduce
        )
        return cls(
            contributions=contributions,
            source_types=tuple(source_types),
            conditions=CURRENT_CONDITIONS,
            target_type=target_type or PRIMARY_TARGET_TYPE,
            arm=arm,
        )


def _trapezoid(y: np.ndarray, dx: float) -> np.ndarray:
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return fn(y, dx=dx, axis=-1)


def _current_trace(view: Any, source_type: str) -> np.ndarray:
    """`cv[source_type][:]` -- flyvis's own accessor, as float64 numpy."""
    try:
        arr = view[source_type][:]
    except Exception as exc:  # pragma: no cover - depends on flyvis at runtime
        raise KeyError(
            f"could not read current trace for source type {source_type!r} from "
            f"the supplied current view ({type(view).__name__}). The "
            f"pre-registered source-type set must exist in every model's view; "
            f"do not silently drop it."
        ) from exc
    return _asnumpy(arr)


def time_integrated_currents(
    cv_pd: Any,
    cv_nd: Any,
    source_types: Sequence[str],
    *,
    dt: float,
    reduce: str = "integral",
) -> np.ndarray:
    """Time-integrate PD/ND current traces into a (n_models, n_types, 2) array.

    Definition
    ----------
    For presynaptic type s and condition k in {PD, ND}, with I_{s,k}(t) the
    current from type s into the target cell at time t (already summed over the
    target's input cells by flyvis's `sum_over_cells`):

        reduce="integral" (default):  C_{s,k} = trapezoid_t I_{s,k}(t) dt
        reduce="sum":                 C_{s,k} = dt * sum_t I_{s,k}(t)
        reduce="mean":                C_{s,k} = mean_t I_{s,k}(t)

    "integral" is what PLAN section 6.1 specifies ("time-integrated signed
    current vector"); the other two exist so a caller can check that the choice
    does not matter (they differ from "integral" only by O(dt) edge terms and a
    constant factor, and L1 normalization removes the constant factor).

    The trace is integrated over its LAST axis (time), then singleton axes are
    squeezed out, leaving one value per model. Signs are preserved: inhibitory
    presynaptic types give negative entries and must not be rectified.

    Parameters
    ----------
    cv_pd, cv_nd : flyvis MovingEdgeCurrentView objects (or anything supporting
        `view[source_type][:]` -> array with time last).
    source_types : fixed, pre-registered order.
    dt : seconds per timestep.
    """
    if reduce not in ("integral", "sum", "mean"):
        raise ValueError("reduce must be 'integral', 'sum' or 'mean'")
    if not (dt > 0):
        raise ValueError("dt must be positive")
    sts = tuple(source_types)
    if not sts:
        raise ValueError("source_types must be non-empty")
    if len(set(sts)) != len(sts):
        raise ValueError("source_types must be unique")

    per_condition = []
    for view in (cv_pd, cv_nd):
        cols = []
        for st in sts:
            tr = _current_trace(view, st)
            if tr.ndim < 1 or tr.shape[-1] < 2:
                raise ValueError(
                    f"current trace for {st!r} has shape {tr.shape}; the last "
                    f"axis must be time with at least 2 samples"
                )
            if reduce == "integral":
                val = _trapezoid(tr, dx=dt)
            elif reduce == "sum":
                val = tr.sum(axis=-1) * dt
            else:
                val = tr.mean(axis=-1)
            cols.append(np.atleast_1d(np.squeeze(val)))
        widths = {c.shape for c in cols}
        if len(widths) != 1:
            raise ValueError(
                f"source types disagree on the number of models: {widths}"
            )
        per_condition.append(np.stack(cols, axis=-1))       # (n_models, n_types)
    out = np.stack(per_condition, axis=-1)                   # (n_models, n_types, 2)
    if out.ndim != 3:
        raise ValueError(
            f"expected (n_models, n_source_types, 2) after integration, got {out.shape}"
        )
    return out


def mechanism_fingerprint(
    contributions: np.ndarray, *, normalize: str = "l1"
) -> np.ndarray:
    """PRIMARY M2 FINGERPRINT: flatten and L1-normalize a current decomposition.

    Definition
    ----------
    With C_{s,k} the time-integrated signed current from presynaptic type s at
    condition k (PD, ND), the fingerprint is the flattened vector

        m = vec(C) / sum_{s,k} |C_{s,k}|      in R^(n_source_types * n_conditions)

    JOINT normalization over the WHOLE source-type x condition vector is the
    pre-registered choice, not per-condition normalization. Normalizing PD and
    ND separately would destroy the PD/ND magnitude asymmetry, which IS the
    direction-selectivity mechanism; joint normalization keeps it and removes
    only the overall current scale.

    Accepts (n_source_types, n_conditions)            -> (S*K,)
         or (n_models, n_source_types, n_conditions)  -> (n_models, S*K).

    `normalize` alternatives ("none", "l2", "max") exist for sensitivity checks
    only. l1 is the pre-registered normalization (`PRIMARY_ANALYSIS`); do not
    shop between them.
    """
    C = _asnumpy(contributions)
    if C.ndim == 2:
        flat = C.reshape(-1)
    elif C.ndim == 3:
        flat = C.reshape(C.shape[0], -1)
    else:
        raise ValueError(
            "contributions must be (n_source_types, n_conditions) or "
            "(n_models, n_source_types, n_conditions)"
        )
    return _normalize_last_axis(flat, normalize)


def mechanism_distance(
    contributions_a: np.ndarray,
    contributions_b: np.ndarray,
    source_types_a: Sequence[str],
    source_types_b: Optional[Sequence[str]] = None,
    *,
    metric: str = "euclidean",
    normalize: str = "l1",
    target_type: str = PRIMARY_TARGET_TYPE,
) -> MechanismDistance:
    """PRIMARY mechanism distance between TWO trained models.

    Definition
    ----------
    Match presynaptic types by NAME (`_match_names`), build each model's
    fingerprint m = `mechanism_fingerprint` over the matched types, then

      metric="euclidean" (PRE-REGISTERED):  D = || m_A - m_B ||_2
      metric="cosine":                      D = (1 - <m_A, m_B>/(||m_A|| ||m_B||)) / 2
      metric="correlation":                 D = (1 - pearson_r(m_A, m_B)) / 2

    Euclidean on jointly-L1-normalized fingerprints is the pre-registered
    choice because (a) it is a TRUE METRIC, which is what PERMDISP's principal
    coordinate step assumes, and (b) the squared distances then decompose
    exactly into `ensemble_total_variance` (metric M2), so the test statistic
    and the reported effect size are two views of one quantity. Correlation
    distance is a squared-chord distance, not a metric, and would introduce
    negative eigenvalues that Anderson's correction then has to absorb.

    `per_source_type[s] = || m_A[s, :] - m_B[s, :] ||_2` is returned for every
    metric as a diagnostic (which presynaptic types drive the difference). For
    metric="euclidean" it is an exact decomposition,
    value = sqrt(sum_s per_source_type[s]^2); for the other metrics it is NOT a
    decomposition of `value` and is reported for interpretation only.
    """
    A = _asnumpy(contributions_a)
    B = _asnumpy(contributions_b)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            "contributions_a/b must be 2-D (n_source_types, n_conditions)"
        )
    if A.shape[-1] != B.shape[-1]:
        raise ValueError(
            "the two models must have the same conditions: "
            f"{A.shape[-1]} vs {B.shape[-1]}"
        )
    sts_b = source_types_a if source_types_b is None else source_types_b
    shared, ia, ib = _match_names(source_types_a, sts_b)

    ma = mechanism_fingerprint(A[ia], normalize=normalize)
    mb = mechanism_fingerprint(B[ib], normalize=normalize)

    if metric == "euclidean":
        value = float(np.sqrt(((ma - mb) ** 2).sum()))
    elif metric == "cosine":
        na, nb = float(np.linalg.norm(ma)), float(np.linalg.norm(mb))
        cos = float(ma @ mb / (na * nb)) if na > 0 and nb > 0 else 0.0
        value = (1.0 - cos) / 2.0
    elif metric == "correlation":
        value = float((1.0 - _pearson_rows(ma[None, :], mb[None, :])[0]) / 2.0)
    else:
        raise ValueError(f"unknown metric {metric!r}")

    n_cond = A.shape[-1]
    diff = (ma - mb).reshape(len(shared), n_cond)
    per_type = np.sqrt((diff ** 2).sum(axis=-1))
    return MechanismDistance(
        value=value,
        per_source_type={s: float(v) for s, v in zip(shared, per_type)},
        metric=metric,
        source_types=shared,
        target_type=target_type,
    )


def mechanism_distance_matrix(
    ensemble: "CurrentEnsemble",
    *,
    metric: str = "euclidean",
    normalize: str = "l1",
) -> np.ndarray:
    """All-pairs PRIMARY mechanism distances within one arm (currents).

    Returns a symmetric (n_models, n_models) matrix with a zero diagonal. This
    is the input to `permdisp`, the study's PRIMARY TEST.

    Raises TypeError on a `TuningEnsemble`: tuning curves are a RESPONSE metric
    and have their own function, `response_distance_matrix`. The two must not
    be reported under one name (design decision D4).
    """
    if isinstance(ensemble, TuningEnsemble):
        raise TypeError(
            "mechanism_distance_matrix() takes a CurrentEnsemble (the PRIMARY "
            "mechanism metric M2). For moving-edge tuning curves use "
            "response_distance_matrix() -- that is a SECONDARY RESPONSE metric, "
            "not a mechanism metric."
        )
    if not isinstance(ensemble, CurrentEnsemble):
        raise TypeError(f"expected a CurrentEnsemble, got {type(ensemble).__name__}")
    fps = mechanism_fingerprint(ensemble.contributions, normalize=normalize)
    return fingerprint_distance_matrix(fps, metric=metric)


def fingerprint_distance_matrix(
    fingerprints: np.ndarray, *, metric: str = "euclidean"
) -> np.ndarray:
    """All-pairs distance matrix between already-built fingerprint vectors.

    metric="euclidean": D_ij = ||m_i - m_j||_2   (a true metric; PRE-REGISTERED)
    metric="cosine" / "correlation": (1 - similarity) / 2, in [0, 1].

    Computed from the Gram matrix, so it is O(N^2 p) rather than O(N^2 p) with a
    Python loop; the smoke test checks it against the pairwise loop.
    """
    M = _asnumpy(fingerprints)
    if M.ndim != 2:
        raise ValueError("fingerprints must be (n_models, n_features)")
    if metric == "euclidean":
        sq = (M ** 2).sum(axis=1)
        D2 = sq[:, None] + sq[None, :] - 2.0 * (M @ M.T)
        np.maximum(D2, 0.0, out=D2)
        D = np.sqrt(D2)
    elif metric in ("cosine", "correlation"):
        X = M - M.mean(axis=1, keepdims=True) if metric == "correlation" else M
        norms = np.sqrt((X ** 2).sum(axis=1))
        safe = np.where(norms > 0, norms, 1.0)
        Xn = X / safe[:, None]
        sim = Xn @ Xn.T
        sim[norms == 0, :] = 0.0
        sim[:, norms == 0] = 0.0
        D = (1.0 - np.clip(sim, -1.0, 1.0)) / 2.0
    else:
        raise ValueError(f"unknown metric {metric!r}")
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2.0


# =============================================================================
# 4. SECONDARY RESPONSE METRIC (R1): MOVING-EDGE ANGULAR TUNING CURVES
# =============================================================================
#
# EVERYTHING IN THIS SECTION IS A RESPONSE METRIC, NOT A MECHANISM METRIC.
# It measures what the network outputs, not what circuit produced it. It is
# retained because (a) it is cheap -- moving-edge responses need no Sintel
# download -- (b) it is the closest available analogue of [Karaneen2026]'s own
# clustering geometry, and (c) a mechanism claim is strengthened if the response
# variance does NOT move in lockstep. It is SECONDARY. Do not report it as
# "mechanism variance" (design decision D4 / reviewer issue science-6).


@dataclass(frozen=True)
class ResponseDistance:
    """Result of a model-to-model RESPONSE (tuning-curve) distance."""

    value: float
    per_cell_type: Dict[str, float]
    metric: str
    cell_types: Tuple[str, ...]


@dataclass(frozen=True)
class TuningEnsemble:
    """Angular tuning curves for an ensemble of independently seeded models.

    Attributes
    ----------
    curves : (n_models, n_cell_types, n_angles) array of peak responses.
    cell_types : names, length n_cell_types, indexing axis 1.
    angles_deg : probe directions, length n_angles, indexing axis 2.
    arm : optional label ("constrained" / "baseline").
    """

    curves: np.ndarray
    cell_types: Tuple[str, ...]
    angles_deg: np.ndarray
    arm: str = ""

    def __post_init__(self) -> None:
        curves = _asnumpy(self.curves)
        angles = _asnumpy(self.angles_deg).ravel()
        cts = tuple(self.cell_types)
        if curves.ndim != 3:
            raise ValueError(
                f"curves must be (n_models, n_cell_types, n_angles); got {curves.shape}"
            )
        if curves.shape[1] != len(cts):
            raise ValueError(
                f"curves axis 1 = {curves.shape[1]} but got {len(cts)} cell types"
            )
        if curves.shape[2] != angles.size:
            raise ValueError(
                f"curves axis 2 = {curves.shape[2]} but got {angles.size} angles"
            )
        if len(set(cts)) != len(cts):
            raise ValueError("cell_types must be unique")
        object.__setattr__(self, "curves", curves)
        object.__setattr__(self, "angles_deg", angles)
        object.__setattr__(self, "cell_types", cts)

    @property
    def n_models(self) -> int:
        return self.curves.shape[0]

    def model(self, i: int) -> np.ndarray:
        """(n_cell_types, n_angles) tuning curves of model i."""
        return self.curves[i]


def _normalize_curves(curves: np.ndarray, how: str) -> np.ndarray:
    """PER-CELL-TYPE normalization applied before a response distance is taken.

    Normalizes each cell type's curve independently (last axis), which strips
    that cell type's overall response gain and keeps the SHAPE of its tuning
    curve. Contrast with `mechanism_fingerprint`, which normalizes the whole
    source-type x condition vector JOINTLY because the relative magnitudes
    across presynaptic types are themselves the mechanism.
    """
    return _normalize_last_axis(curves, how)


def tuning_curve_response_distance(
    curves_a: np.ndarray,
    curves_b: np.ndarray,
    cell_types_a: Sequence[str],
    cell_types_b: Optional[Sequence[str]] = None,
    *,
    metric: str = "correlation",
    normalize: str = "l1",
    weights: Optional[Mapping[str, float]] = None,
) -> ResponseDistance:
    """SECONDARY: tuning-curve RESPONSE distance between TWO trained models.

    THIS IS NOT A MECHANISM METRIC (design decision D4). It was called
    `tuning_curve_mechanism_distance` before round 2; the rename is the point.
    Angular tuning curves are the model's task-relevant OUTPUT, so two networks
    with different circuits and identical T4c tuning are indistinguishable here
    -- which is precisely the degeneracy the study is about. The primary
    mechanism metric is `mechanism_distance` (current decomposition).

    Definition
    ----------
    Let C be the set of cell types the two models share (see `_match_names`),
    and let r_c^(m)(theta) be model m's angular tuning curve for cell type c,
    optionally normalized per cell type (`normalize`). Per cell type:

      metric="correlation" (default):
          d_c = (1 - pearson_r_theta( r_c^(A), r_c^(B) )) / 2            in [0, 1]
      metric="cosine":
          d_c = (1 - <r_c^(A), r_c^(B)> / (||r_c^(A)|| ||r_c^(B)||)) / 2 in [0, 1]
      metric="euclidean":
          d_c = || r_c^(A) - r_c^(B) ||_2                               in [0, inf)

    and the model-to-model distance is the (optionally weighted) mean over cell
    types:

          D(A, B) = sum_c w_c d_c / sum_c w_c ,   w_c = 1 by default.

    Properties: for "correlation"/"cosine", D is symmetric, D(A, A) = 0, and
    D is bounded in [0, 1]; it is NOT a metric in the triangle-inequality sense
    (correlation distance is a squared-chord distance -- take sqrt(2*d_c) if you
    need a true metric, e.g. before a PCoA-based test). "euclidean" with
    normalize="l1" IS a true metric and is what `permdisp` expects for an exact
    PERMDISP interpretation.

    Why this definition
    -------------------
    [Karaneen2026] measures model-to-model similarity through UMAP with
    `metric="correlation"` on per-model responses, then a Gaussian mixture.
    Correlation distance is therefore the geometry the paper's own clustering
    is built on, and this function keeps it while removing the two pieces that
    cannot enter a variance test: the stochastic UMAP embedding and the
    BIC-selected mixture model.

    APPROXIMATION -- NOT A REPRODUCTION OF THE PAPER'S METRIC
    ---------------------------------------------------------
    The paper computes no such distance. Concretely, this function differs from
    the paper's clustering pipeline in four ways, all deliberate:
      (1) STIMULUS: we use moving-edge angular tuning curves; the paper's
          clustering uses responses to NATURALISTIC stimuli (AugmentedSintel,
          dt=1/100, central-column activity only). Naturalistic responses need
          the multi-GB Sintel download; moving-edge responses do not.
      (2) CELL TYPES: we compare an explicit, matched set (default the eight
          T4/T5 subtypes); the paper's clustering uses ONE cell type (T4c).
      (3) NO UMAP: the paper embeds with UMAP
          (min_dist=0.105, spread=9.0, n_neighbors=5, random_state=42,
          n_epochs=1500, n_components=2, metric="correlation") before
          clustering. UMAP is itself seed-dependent, so it cannot sit inside a
          reproducibility measurement without confounding it.
      (4) NO CLUSTERING: no Gaussian mixture, no BIC model selection, no
          task-error-sorted cluster labels.
    If you need the paper's actual figure, call flyvis
    (`Ensemble.clustering(cell_type)` / `cluster_indices`). Use THIS function
    when you need a RESPONSE scalar that a variance test can consume.

    Returns
    -------
    ResponseDistance(value, per_cell_type, metric, cell_types)
    """
    curves_a = _asnumpy(curves_a)
    curves_b = _asnumpy(curves_b)
    if curves_a.ndim != 2 or curves_b.ndim != 2:
        raise ValueError("curves_a/curves_b must be 2-D (n_cell_types, n_angles)")
    if curves_a.shape[-1] != curves_b.shape[-1]:
        raise ValueError(
            "the two models must be probed at the same angles: "
            f"{curves_a.shape[-1]} vs {curves_b.shape[-1]}"
        )
    cts_b = cell_types_a if cell_types_b is None else cell_types_b
    shared, ia, ib = _match_names(cell_types_a, cts_b)

    a = _normalize_curves(curves_a[ia], normalize)
    b = _normalize_curves(curves_b[ib], normalize)

    if metric == "correlation":
        d = (1.0 - _pearson_rows(a, b)) / 2.0
    elif metric == "cosine":
        na = np.sqrt((a * a).sum(-1))
        nb = np.sqrt((b * b).sum(-1))
        denom = na * nb
        cos = np.zeros_like(denom)
        good = denom > 0
        cos[good] = (a * b).sum(-1)[good] / denom[good]
        d = (1.0 - cos) / 2.0
    elif metric == "euclidean":
        d = np.sqrt(((a - b) ** 2).sum(-1))
    else:
        raise ValueError(f"unknown metric {metric!r}")

    if weights is None:
        w = np.ones(len(shared))
    else:
        w = np.array([float(weights.get(ct, 1.0)) for ct in shared])
        if np.all(w == 0):
            raise ValueError("all weights are zero")
    value = float((w * d).sum() / w.sum())
    return ResponseDistance(
        value=value,
        per_cell_type={ct: float(v) for ct, v in zip(shared, d)},
        metric=metric,
        cell_types=shared,
    )


def response_fingerprint(curves: np.ndarray, *, normalize: str = "l1") -> np.ndarray:
    """SECONDARY: flatten per-model tuning curves into one RESPONSE fingerprint.

    Definition: normalize each cell type's curve (default L1, which strips that
    cell type's overall response gain so the statistic measures curve SHAPE
    rather than output scale), then concatenate across cell types:

        r = concat_c ( r_c / ||r_c||_1 )   in R^(n_cell_types * n_angles)

    Accepts (n_cell_types, n_angles) -> (n_cell_types*n_angles,)
         or (n_models, n_cell_types, n_angles) -> (n_models, n_cell_types*n_angles).

    RENAMED from `mechanism_fingerprint` in round 2. That name now belongs to
    the current-decomposition fingerprint, which is the primary M2; a tuning
    curve is an output, not a mechanism (design decision D4).

    NOTE ON DIMENSION. With the default eight T4/T5 subtypes at 12 angles this
    is a 96-dimensional vector -- the setting in which Brown-Forsythe on
    medoid deviations collapses (see `medoid_leverage` and `deviations_from_center`).
    """
    curves = _asnumpy(curves)
    normed = _normalize_curves(curves, normalize)
    if curves.ndim == 2:
        return normed.reshape(-1)
    if curves.ndim == 3:
        return normed.reshape(curves.shape[0], -1)
    raise ValueError("curves must be 2-D or 3-D")


def response_distance_matrix(
    ensemble: TuningEnsemble,
    *,
    metric: str = "correlation",
    normalize: str = "l1",
    weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """SECONDARY: all-pairs RESPONSE distances within one ensemble.

    Returns a symmetric (n_models, n_models) matrix with a zero diagonal, where
    entry (i, j) is `tuning_curve_response_distance(model_i, model_j).value`.
    Feed it to the same `permdisp` the primary analysis uses -- but report it as
    a response result, never as a mechanism result (design decision D4).
    """
    if isinstance(ensemble, CurrentEnsemble):
        raise TypeError(
            "response_distance_matrix() takes a TuningEnsemble. For the PRIMARY "
            "mechanism metric (current decomposition) use mechanism_distance_matrix()."
        )
    n = ensemble.n_models
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = tuning_curve_response_distance(
                ensemble.curves[i], ensemble.curves[j],
                ensemble.cell_types, ensemble.cell_types,
                metric=metric, normalize=normalize, weights=weights,
            ).value
            D[i, j] = D[j, i] = d
    return D


# =============================================================================
# 5. ACROSS-SEED VARIANCE / DISPERSION FOR AN ENSEMBLE OF N MODELS
# =============================================================================


def ensemble_total_variance(fingerprints: np.ndarray) -> float:
    """Across-seed TOTAL VARIANCE of an ensemble of mechanism fingerprints.

    Definition
    ----------
    For fingerprints m_1..m_N in R^p with centroid mbar = (1/N) sum_s m_s:

        V = (1/N) sum_s ||m_s - mbar||_2^2
          = trace( Cov(m) )                       [population covariance, ddof=0]
          = (1 / (2 N^2)) sum_i sum_j ||m_i - m_j||_2^2

    All three expressions are algebraically identical (the smoke test asserts
    agreement to 1e-10). The pairwise form is what generalizes to a non-Euclidean
    distance and is the quantity `permdisp` tests.

    Applies to EITHER fingerprint. Fed `mechanism_fingerprint` output it is
    metric M2, the study's primary effect size ("mechanism-fingerprint total
    variance"); fed `response_fingerprint` output it is the secondary R1
    dispersion. The function cannot tell which it was given -- say which in the
    report. It is OUR definition, not the paper's: [Karaneen2026] reports no
    variance statistic of any kind. Its role is to turn "models trained with
    different seeds implement different mechanisms" into one number per arm.

    ddof NOTE: this is the POPULATION trace (1/N, ddof=0), chosen because it is
    the quantity with the exact pairwise-distance identity above and the exact
    correspondence to PERMDISP's centroid dispersions (mean of z^2 = V). Every
    other variance in this module uses ddof=1. For a RATIO between arms with
    equal N the choice cancels; say which you report.

    Units: squared units of the (normalized) fingerprint. Only ratios between
    arms are interpretable; report log(V_A / V_B) with a CI, not V alone.
    """
    M = _asnumpy(fingerprints)
    if M.ndim != 2:
        raise ValueError("fingerprints must be (n_models, n_features)")
    if M.shape[0] < 2:
        raise ValueError("need at least 2 models")
    centered = M - M.mean(axis=0, keepdims=True)
    return float((centered ** 2).sum(axis=1).mean())


def deviations_from_center(
    D: np.ndarray, *, center: str = "mean_pairwise"
) -> np.ndarray:
    """SECONDARY: per-model scalar deviation from the ensemble center.

    A univariate summary, one number per seed, so that a SCALAR two-arm test
    (Brown-Forsythe, permutation) can be run on a distance-based endpoint.

    Definition
    ----------
      center="mean_pairwise" (default since round 2): d_s = (1/(N-1))
        sum_{j != s} D[s, j], the mean distance from model s to every other
        model. No reference model, no exact zero. The values are NOT
        independent (each d_s shares terms with every other), so the nominal
        F reference distribution is still only approximate -- use the
        permutation companion, or better, use `permdisp`.
      center="medoid": let k = argmin_i sum_j D[i, j] and return d_s = D[s, k].
        RETAINED FOR THE RECORD AND FOR THE MISCALIBRATION DEMONSTRATION ONLY.

    WHY center="medoid" + BROWN-FORSYTHE IS NOT THE PRIMARY TEST (design
    decision D3 / reviewer issue science-3)
    --------------------------------------------------------------------------
    The medoid is a member of its own arm, so d_k = 0 EXACTLY, for one model per
    arm. In Brown-Forsythe the deviations are re-centred on the median,
    z_s = |d_s - median(d)|, so that exact zero becomes the LARGEST z in the
    arm -- z_k = median(d), while every other z_s is a fluctuation about it. In
    high dimension pairwise distances concentrate (all d_s approximately equal),
    so the z_s for s != k are small and the single z_k dominates Levene's
    denominator sum_s (z_s - zbar)^2. That denominator term is nearly identical
    in both arms and nearly identical under label permutation, so it acts as a
    constant that crushes the statistic: the test's size falls far below
    nominal and its power at a real difference in dispersion collapses with it.
    `medoid_leverage` measures the share; the smoke test prints it for the
    96-dimensional response fingerprint and for the primary current
    fingerprint, together with the resulting size and power.

    THE FIX IS NOT A DIFFERENT SCALAR TEST. A rank test on the same deviations
    is worse (the d_s are dependent, so its FPR is badly inflated). The fix is
    to test the multivariate dispersion directly with a label permutation, i.e.
    `permdisp`, which is the study's primary test.
    """
    D = _asnumpy(D)
    n = D.shape[0]
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix")
    if center == "medoid":
        k = int(np.argmin(D.sum(axis=1)))
        return D[:, k].copy()
    if center == "mean_pairwise":
        return (D.sum(axis=1) / (n - 1)).copy()
    raise ValueError(f"unknown center {center!r}")


def medoid_leverage(D: np.ndarray) -> Dict[str, float]:
    """Quantify the medoid's leverage on the Brown-Forsythe denominator.

    With d = `deviations_from_center(D, center="medoid")`, k the medoid index
    and z_s = |d_s - median(d)| the Brown-Forsythe deviations, returns

        share  = (z_k - zbar)^2 / sum_s (z_s - zbar)^2

    i.e. the fraction of one arm's contribution to Levene's WITHIN sum of
    squares that comes from the medoid's own (exactly zero) distance, plus the
    supporting quantities. A share near 1 means the test statistic is a ratio
    dominated by a single deterministic term and can no longer respond to the
    arm's actual dispersion. This is the diagnostic behind design decision D3.

    Returns keys: `medoid_index`, `share_of_within_ss`, `z_medoid`,
    `z_mean_others`, `median_distance`, `cv_of_distances`, `n`.
    """
    D = _asnumpy(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix")
    n = D.shape[0]
    k = int(np.argmin(D.sum(axis=1)))
    d = D[:, k]
    z = np.abs(d - np.median(d))
    zbar = float(z.mean())
    ss = float(((z - zbar) ** 2).sum())
    others = np.delete(d, k)
    return {
        "medoid_index": float(k),
        "share_of_within_ss": float(((z[k] - zbar) ** 2) / ss) if ss > 0 else float("nan"),
        "z_medoid": float(z[k]),
        "z_mean_others": float(np.delete(z, k).mean()),
        "median_distance": float(np.median(d)),
        "cv_of_distances": float(others.std(ddof=1) / others.mean()) if others.mean() > 0 else float("nan"),
        "n": float(n),
    }


def circular_dispersion(theta_rad: np.ndarray) -> Dict[str, float]:
    """Across-seed dispersion of a preferred direction (circular statistics).

    Definition (metric M1 of the study design)
    ------------------------------------------
    For preferred directions theta_1..theta_N (radians):

        Rbar      = | (1/N) sum_s exp(1j * theta_s) |          in [0, 1]
        V_circ    = 1 - Rbar                                    in [0, 1]
        sigma_circ = sqrt(-2 ln Rbar)                            in [0, inf)
        mean_dir  = angle( sum_s exp(1j * theta_s) )

    V_circ = 0 means every seed found the same preferred direction; V_circ -> 1
    means the directions are spread uniformly around the circle. sigma_circ is
    the circular standard deviation (Mardia), in radians, and is the quantity
    comparable in magnitude to a linear SD for concentrated data.

    Reference: Mardia KV & Jupp PE, Directional Statistics (2000), sec. 2.3.
    Use with preferred directions from `preferred_direction` -- flyvis scores T4
    at intensity=1 and T5 at intensity=0.

    This is the cheapest of the three mechanism-variance metrics and has the
    strongest biological anchor (KNOWN_PREFERRED_DIRECTIONS_DEG). Pre-register
    T4c as the primary cell type -- it is the one both [Lappalainen2024] and
    [Karaneen2026] cluster on -- and Holm-correct if you report all eight.
    """
    theta = _check_1d(theta_rad, "theta_rad")
    z = np.exp(1j * theta)
    Rbar = float(np.abs(z.mean()))
    Rbar_clipped = min(max(Rbar, 1e-12), 1.0)
    return {
        "R_bar": Rbar,
        "circular_variance": 1.0 - Rbar,
        "circular_sd_rad": float(math.sqrt(-2.0 * math.log(Rbar_clipped))),
        "circular_sd_deg": float(math.degrees(math.sqrt(-2.0 * math.log(Rbar_clipped)))),
        "mean_direction_rad": float(np.angle(z.sum())),
        "mean_direction_deg": float(np.rad2deg(np.angle(z.sum())) % 360.0),
        "n": int(theta.size),
    }


def circular_deviation_from_median(theta_rad: np.ndarray) -> np.ndarray:
    """Per-seed absolute angular deviation from the ensemble's circular median.

    Definition
    ----------
        theta_med = argmin_{theta_s} sum_j | arg( exp(1j*(theta_j - theta_s)) ) |
                    (circular median, restricted to observed directions)
        a_s       = | arg( exp(1j*(theta_s - theta_med)) ) |   in [0, pi]

    The a_s are the per-seed scalars to feed `brown_forsythe_variance_test`
    (equivalently, a hand-rolled Wallraff test: R's `circular::wallraff.test`
    is a rank-sum test on exactly these angular deviations). Returns radians.
    """
    theta = _check_1d(theta_rad, "theta_rad")
    pairwise = np.abs(np.angle(np.exp(1j * (theta[:, None] - theta[None, :]))))
    med = theta[int(np.argmin(pairwise.sum(axis=1)))]
    return np.abs(np.angle(np.exp(1j * (theta - med))))


@dataclass(frozen=True)
class EnsembleDispersion:
    """Across-seed dispersion summary for one arm.

    `kind` says WHICH claim this dispersion supports: "mechanism (currents)" is
    the primary M2 endpoint; "response (tuning curves)" is the secondary R1
    output metric. They are never interchangeable (design decision D4).
    """

    n_models: int
    total_variance: float
    mean_pairwise_distance: float
    median_pairwise_distance: float
    deviations: np.ndarray           # per-seed scalar, SECONDARY route only
    distance_matrix: np.ndarray = field(repr=False)
    metric: str = "euclidean"
    arm: str = ""
    kind: str = "mechanism (currents)"
    n_features: int = 0

    def summary(self) -> str:
        return (
            f"arm={self.arm or '?'} kind={self.kind} N={self.n_models} "
            f"p={self.n_features} "
            f"total_variance={self.total_variance:.5g} "
            f"mean_pairwise_d={self.mean_pairwise_distance:.5g} "
            f"sd(deviations)={float(np.std(self.deviations, ddof=1)):.5g}"
        )


def across_seed_dispersion(
    ensemble: Any,
    *,
    metric: Optional[str] = None,
    normalize: str = "l1",
    center: str = "mean_pairwise",
    weights: Optional[Mapping[str, float]] = None,
) -> EnsembleDispersion:
    """One-call across-seed dispersion for an ensemble of N trained models.

    Dispatches on the ensemble type and LABELS the result accordingly:
      * `CurrentEnsemble` -> PRIMARY mechanism metric M2 (current
        decomposition), default metric "euclidean";
      * `TuningEnsemble`  -> SECONDARY response metric R1 (tuning curves),
        default metric "correlation".

    Computes:
      * `total_variance` -- `ensemble_total_variance` of the L1-normalized
        fingerprints (Euclidean, gain-free). For the primary configuration
        (euclidean + l1) this is exactly the quantity `permdisp` partitions.
      * `mean_pairwise_distance` / `median_pairwise_distance` -- from the
        distance matrix.
      * `deviations` -- per-seed scalar summary, for the SECONDARY scalar tests
        only. The primary test (`permdisp`) consumes `distance_matrix`, not
        this vector.

    Note the two families are deliberately not forced to agree: total variance
    is always Euclidean-on-fingerprints, while the distance matrix may be
    correlation-based for the secondary response metric. Report whichever you
    pre-registered (`PRIMARY_ANALYSIS`); do not shop between them.
    """
    if isinstance(ensemble, CurrentEnsemble):
        metric = "euclidean" if metric is None else metric
        if weights is not None:
            raise ValueError("weights are not defined for the current-decomposition metric")
        D = mechanism_distance_matrix(ensemble, metric=metric, normalize=normalize)
        fps = mechanism_fingerprint(ensemble.contributions, normalize=normalize)
        kind = "mechanism (currents)"
    elif isinstance(ensemble, TuningEnsemble):
        metric = "correlation" if metric is None else metric
        D = response_distance_matrix(
            ensemble, metric=metric, normalize=normalize, weights=weights
        )
        fps = response_fingerprint(ensemble.curves, normalize=normalize)
        kind = "response (tuning curves)"
    else:
        raise TypeError(
            f"expected a CurrentEnsemble or TuningEnsemble, got {type(ensemble).__name__}"
        )
    iu = np.triu_indices(ensemble.n_models, k=1)
    return EnsembleDispersion(
        n_models=ensemble.n_models,
        total_variance=ensemble_total_variance(fps),
        mean_pairwise_distance=float(D[iu].mean()),
        median_pairwise_distance=float(np.median(D[iu])),
        deviations=deviations_from_center(D, center=center),
        distance_matrix=D,
        metric=metric,
        arm=ensemble.arm,
        kind=kind,
        n_features=int(fps.shape[1]),
    )


# ---------------------------------------------------------------------------
# PERMDISP -- multivariate homogeneity of dispersions.  THE PRIMARY TEST (D3).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PermdispResult:
    statistic: float
    pvalue: float
    n_perm: int
    group_dispersions: Dict[Any, float]
    center: str
    method: str = "PERMDISP (Anderson 2006), PCoA + permutation"

    def summary(self) -> str:
        disp = ", ".join(f"{k}={v:.5g}" for k, v in self.group_dispersions.items())
        return (
            f"PERMDISP(center={self.center}): F={self.statistic:.4f} "
            f"p={self.pvalue:.4g} ({self.n_perm} perms) mean z: {disp}"
        )


def _pcoa(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Principal coordinates of a distance matrix.

    Gower double-centring: A = -0.5 * D^2 ; G = (I - 11'/n) A (I - 11'/n).
    Returns (real_coords, imag_coords) for positive and negative eigenvalues
    respectively; [Anderson2006] keeps both, because a non-Euclidean D yields
    negative eigenvalues whose axes are imaginary.
    """
    D = _asnumpy(D)
    n = D.shape[0]
    A = -0.5 * (D ** 2)
    J = np.eye(n) - np.ones((n, n)) / n
    G = J @ A @ J
    G = (G + G.T) / 2.0
    vals, vecs = np.linalg.eigh(G)
    tol = max(1e-10, 1e-10 * np.max(np.abs(vals)))
    pos = vals > tol
    neg = vals < -tol
    real = vecs[:, pos] * np.sqrt(vals[pos])
    imag = vecs[:, neg] * np.sqrt(-vals[neg])
    return real, imag


def _spatial_median(X: np.ndarray, *, max_iter: int = 200, tol: float = 1e-9) -> np.ndarray:
    """Weiszfeld iteration for the spatial (L1) median of rows of X."""
    if X.shape[1] == 0:
        return np.zeros(0)
    y = X.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(X - y, axis=1)
        nz = d > 1e-12
        if not np.any(nz):
            return y
        w = 1.0 / d[nz]
        y_new = (X[nz] * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < tol:
            return y_new
        y = y_new
    return y


def _dispersions(real: np.ndarray, imag: np.ndarray, groups: np.ndarray, center: str) -> np.ndarray:
    """Distance of each object to its own group's centre, Anderson's correction."""
    z = np.empty(real.shape[0], dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        if center == "centroid":
            cr = real[m].mean(axis=0) if real.shape[1] else np.zeros(0)
            ci = imag[m].mean(axis=0) if imag.shape[1] else np.zeros(0)
        elif center == "median":
            cr = _spatial_median(real[m]) if real.shape[1] else np.zeros(0)
            ci = _spatial_median(imag[m]) if imag.shape[1] else np.zeros(0)
        else:
            raise ValueError(f"unknown center {center!r}")
        sq_real = ((real[m] - cr) ** 2).sum(axis=1) if real.shape[1] else 0.0
        sq_imag = ((imag[m] - ci) ** 2).sum(axis=1) if imag.shape[1] else 0.0
        # Anderson 2006: real and imaginary contributions SUBTRACT.
        z[m] = np.sqrt(np.abs(sq_real - sq_imag))
    return z


def _gower(D: np.ndarray) -> np.ndarray:
    """Gower double-centred matrix G = J (-0.5 D^2) J, J = I - 11'/n."""
    D = _asnumpy(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix")
    A = -0.5 * (D ** 2)
    rm = A.mean(axis=1, keepdims=True)
    cm = A.mean(axis=0, keepdims=True)
    G = A - rm - cm + A.mean()
    return (G + G.T) / 2.0


def _dispersions_centroid_from_gram(G: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Anderson's z to the GROUP CENTROID, computed directly from the Gram matrix.

    For a group g with membership indicator vector e (n_g ones), PCoA places
    object s at coordinates whose signed squared distance to the group centroid
    is the quadratic form

        q_s = G[s,s] - 2 * mean_{j in g} G[s,j] + mean_{i,j in g} G[i,j],

    and Anderson's imaginary-axis correction is z_s = sqrt(|q_s|). This is
    algebraically identical to eigendecomposing G, splitting the axes by the
    sign of the eigenvalue and subtracting the imaginary part from the real
    part -- but it needs no eigendecomposition, which is what makes the
    end-to-end power simulation in section 7 affordable. `_dispersions` (the
    literal PCoA transcription) is kept and the smoke test asserts the two agree
    to 1e-10 on both a Euclidean and a non-Euclidean distance matrix.

    Parameters
    ----------
    G : (n, n) Gower-centred matrix, from `_gower`.
    member : (k, n) 0/1 membership matrix, one row per group (rows sum to n_g).

    Returns
    -------
    (n,) array of z values (each object measured against ITS OWN group).
    """
    counts = member.sum(axis=1, keepdims=True)                  # (k, 1)
    cross = member @ G                                          # (k, n)
    mean_sj = cross / counts                                    # (k, n)
    gbar = (cross * member).sum(axis=1, keepdims=True) / (counts ** 2)  # (k, 1)
    q = np.diag(G)[None, :] - 2.0 * mean_sj + gbar              # (k, n)
    return np.sqrt(np.abs((q * member).sum(axis=0)))            # (n,)


def _anova_f_rows(z: np.ndarray, member: np.ndarray) -> np.ndarray:
    """One-way ANOVA F for many permutations at once.

    z : (m, n) values, one row per permutation. member : (m, k, n) 0/1.
    Returns (m,) F statistics with k = member.shape[1] groups.
    """
    counts = member.sum(axis=2)                                  # (m, k)
    sums = (member * z[:, None, :]).sum(axis=2)                  # (m, k)
    means = sums / counts
    n = z.shape[1]
    k = member.shape[1]
    grand = z.mean(axis=1, keepdims=True)
    ss_between = (counts * (means - grand) ** 2).sum(axis=1)
    within = z[:, None, :] - means[:, :, None]
    ss_within = ((within ** 2) * member).sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ss_between / (k - 1)) / (ss_within / (n - k))
    return np.where(np.isfinite(F), F, np.inf)


def _permdisp_centroid_fast(
    D: np.ndarray,
    groups: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[float, float, Dict[Any, float]]:
    """Exact vectorized PERMDISP with center='centroid'. See `permdisp`."""
    labels = np.unique(groups)
    k = labels.size
    n = groups.size
    G = _gower(D)
    member_obs = np.stack([(groups == lab).astype(np.float64) for lab in labels])
    z_obs = _dispersions_centroid_from_gram(G, member_obs)
    f_obs = float(_anova_f_rows(z_obs[None, :], member_obs[None, :, :])[0])

    # Permute the ARM LABELS (this is what handles the dependence between the
    # z_s: they all share the same estimated centres, in every permutation).
    perm = np.argsort(rng.random((n_perm, n)), axis=1)
    sizes = [int((groups == lab).sum()) for lab in labels]
    member = np.zeros((n_perm, k, n), dtype=np.float64)
    start = 0
    for gi, size in enumerate(sizes):
        idx = perm[:, start:start + size]
        np.put_along_axis(member[:, gi, :], idx, 1.0, axis=1)
        start += size
    # z under each permutation: (n_perm, n)
    counts = member.sum(axis=2)                                   # (P, k)
    cross = member @ G                                            # (P, k, n)
    mean_sj = cross / counts[:, :, None]
    gbar = (cross * member).sum(axis=2) / (counts ** 2)           # (P, k)
    q = np.diag(G)[None, None, :] - 2.0 * mean_sj + gbar[:, :, None]
    z_perm = np.sqrt(np.abs((q * member).sum(axis=1)))            # (P, n)
    f_perm = _anova_f_rows(z_perm, member)
    pvalue = float((np.sum(f_perm >= f_obs) + 1) / (n_perm + 1))
    disp = {
        (lab.item() if hasattr(lab, "item") else lab): float(z_obs[groups == lab].mean())
        for lab in labels
    }
    return f_obs, pvalue, disp


def _anova_f(values: np.ndarray, groups: np.ndarray) -> float:
    labels = np.unique(groups)
    k = labels.size
    n = values.size
    grand = values.mean()
    ss_between = 0.0
    ss_within = 0.0
    for g in labels:
        v = values[groups == g]
        ss_between += v.size * (v.mean() - grand) ** 2
        ss_within += ((v - v.mean()) ** 2).sum()
    if ss_within <= 0:
        return float("inf")
    return float((ss_between / (k - 1)) / (ss_within / (n - k)))


def permdisp(
    D: np.ndarray,
    groups: Sequence[Any],
    *,
    center: str = "centroid",
    n_perm: int = 9999,
    rng: Optional[np.random.Generator] = None,
) -> PermdispResult:
    """PRIMARY TEST: multivariate homogeneity of dispersions between arms.

    Definition
    ----------
    [Anderson2006]. Embed the distance matrix D by principal coordinate analysis
    (Gower double-centring + eigendecomposition, keeping both positive and
    negative eigenvalue axes). For each object s in group g compute its distance
    z_s to the group's centre (centroid, or spatial median for `center="median"`),
    with Anderson's correction for imaginary axes

        z_s = sqrt( | ||x_s - c_g||^2_real  -  ||y_s - c_g||^2_imag | ).

    The statistic is the one-way ANOVA F on the z_s across groups. Its null
    distribution is obtained by permuting the group labels (`n_perm` times) and
    recomputing centres and z from the SAME principal coordinates:

        p = (#{F_perm >= F_obs} + 1) / (n_perm + 1).

    WHY THIS IS THE PRIMARY TEST (design decision D3 / issue science-3)
    ------------------------------------------------------------------
    The study's endpoint is the dispersion of a MULTIVARIATE fingerprint, and
    every scalar summary of it (distance to medoid, mean pairwise distance)
    produces N numbers per arm that are NOT independent: they share the same
    estimated centre. A scalar test that assumes independence is therefore
    miscalibrated on this endpoint -- Brown-Forsythe on medoid deviations sits
    far below its nominal size and has almost no power (see
    `deviations_from_center`, `medoid_leverage`, and the smoke test's measured
    table), while a rank test on the same deviations is inflated instead.
    PERMDISP does not assume independence: it re-estimates the centres inside
    every permutation, so the dependence is part of the null distribution. With
    exchangeable seeds and equal arm sizes the label permutation makes the test
    exact by construction, which the smoke test verifies empirically on null
    data run through the ACTUAL pipeline.

    CENTRE: `center="centroid"` is the PRE-REGISTERED choice (`PRIMARY_ANALYSIS`),
    not vegan::betadisper's default of "median". Reasons, in order: (1) with the
    pre-registered Euclidean distance on L1-normalized fingerprints the centroid
    dispersions satisfy mean_s z_s^2 = `ensemble_total_variance` EXACTLY, so the
    test statistic and the reported effect size are two views of one quantity;
    (2) it has an exact closed form from the Gower matrix
    (`_dispersions_centroid_from_gram`), which is what makes the end-to-end
    power simulation in section 7 affordable, and which removes the
    eigendecomposition as a source of implementation risk; (3) its null
    calibration is measured, not assumed (smoke test section [9]).
    `center="median"` (spatial median, Weiszfeld) is available and reported as a
    robustness check; it uses the PCoA path.

    VALIDATION STATUS. This is a from-scratch implementation (the module is
    numpy/scipy/torch only, so `skbio.stats.distance.permdisp` and
    `vegan::betadisper` are not importable here). What HAS been checked, in the
    smoke test: the Gower/quadratic-form dispersions agree with the literal PCoA
    transcription to 1e-10 on Euclidean AND on non-Euclidean (correlation)
    distance matrices; for Euclidean distances the z_s equal plain
    ||x_s - centroid|| computed in the original feature space; the observed F
    equals `scipy.stats.f_oneway` on the same z; and the permutation p-value is
    calibrated on null data. What has NOT been checked: a numerical match to
    vegan::betadisper or skbio on a shared dataset. Implementations differ in
    where they take the absolute value in Anderson's imaginary-axis correction,
    so check one dataset against vegan before a publication claims exact
    equivalence.

    Assumes D is a true metric for a strict interpretation. Correlation distance
    is a squared-chord distance, not a metric; pass `metric="euclidean"`
    distances (or sqrt(2*d) of correlation distances) for a clean reading, or
    accept the negative eigenvalues that Anderson's correction is there to
    absorb.
    """
    D = _asnumpy(D)
    if _torch is not None and isinstance(groups, _torch.Tensor):
        groups = groups.detach().to("cpu").numpy()
    g = np.asarray(groups)
    if D.shape[0] != g.size:
        raise ValueError("len(groups) must equal D.shape[0]")
    if np.unique(g).size < 2:
        raise ValueError("need at least 2 groups")
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    rng = np.random.default_rng() if rng is None else rng

    if center == "centroid":
        f_obs, pvalue, disp = _permdisp_centroid_fast(D, g, n_perm, rng)
        return PermdispResult(
            statistic=f_obs,
            pvalue=pvalue,
            n_perm=n_perm,
            group_dispersions=disp,
            center=center,
            method="PERMDISP (Anderson 2006), Gower quadratic form + permutation",
        )
    if center != "median":
        raise ValueError(f"unknown center {center!r}; use 'centroid' or 'median'")

    real, imag = _pcoa(D)
    z_obs = _dispersions(real, imag, g, center)
    f_obs = _anova_f(z_obs, g)

    count = 0
    for _ in range(n_perm):
        gp = rng.permutation(g)
        z = _dispersions(real, imag, gp, center)
        if _anova_f(z, gp) >= f_obs:
            count += 1
    labels = np.unique(g)
    return PermdispResult(
        statistic=f_obs,
        pvalue=(count + 1) / (n_perm + 1),
        n_perm=n_perm,
        group_dispersions={
            (lab.item() if hasattr(lab, "item") else lab): float(z_obs[g == lab].mean())
            for lab in labels
        },
        center=center,
    )


# =============================================================================
# 6. CORRESPONDENCE TO BIOLOGY
# =============================================================================

@dataclass(frozen=True)
class DSICorrespondence:
    """Per-cell-type comparison of model DSI against the literature target."""

    cell_types: Tuple[str, ...]
    model_dsi: np.ndarray
    target_dsi: np.ndarray
    absolute_error: np.ndarray
    mean_absolute_error: float
    pearson_r: float
    convention: str

    def as_dict(self) -> Dict[str, float]:
        return {ct: float(v) for ct, v in zip(self.cell_types, self.model_dsi)}


def dsi_correspondence(
    model_dsi: np.ndarray,
    cell_types: Sequence[str] = T4T5_CELL_TYPES,
    *,
    convention: str = "vector_sum",
) -> DSICorrespondence:
    """Compare a model's T4/T5 DSI to the measured ground truth.

    Definition
    ----------
    With d_c the model DSI for cell type c and g_c the target:

        AE_c = |d_c - g_c|
        MAE  = mean_c AE_c
        r    = pearson_r over cell types between (d_c) and (g_c)

    Targets
    -------
    convention="vector_sum"  -> KNOWN_DSI_VECTOR_SUM (derived by applying the
        flyvis vector-sum formula to the [Maisak2013] curves; mean 0.397, range
        0.25-0.60). USE THIS with `direction_selectivity_index`.
    convention="classic"     -> KNOWN_DSI_CLASSIC (same curves, classic formula;
        range 0.48-0.92). USE THIS with `direction_selectivity_index_classic`,
        and compare to LITERATURE_DSI_CLASSIC for measured values with n.

    HONEST WARNINGS
    ---------------
    1. Both target dicts are DERIVED from the [Maisak2013] population-average
       tuning curves, NOT read off a table of measured DSIs. A measured
       population DSI with an n does not exist as a single number: the value
       depends on indicator (GCaMP6f ~0.8-0.9 vs ArcLight ~0.4 in the SAME
       cells, PMC10082464), stimulus (gratings vs edges vs bars), speed (DS
       decreases with velocity) and formula (three incompatible ones are in
       use). See LITERATURE_DSI_CLASSIC.
    2. Because of (1), treat DSI MAGNITUDE as a WEAKER correspondence target
       than preferred direction (`preferred_direction_correspondence`) or curve
       shape (correlation to `known_tuning_curves`). The paper-facing statistic
       is `dsi_correlation_to_known`, which only needs the ordering.
    3. Individual T4/T5 cells are not strictly cardinal. Henning et al. 2022
       (Sci Adv 8:eabi7112) report the population clusters around SIX
       directions, not four, and state verbatim that "individual tuning differs
       from the overall average orthogonal tuning of these layers". The
       cardinal targets are a population-average idealization.
    """
    model = _asnumpy(model_dsi).ravel()
    cts = tuple(cell_types)
    if model.size != len(cts):
        raise ValueError(f"model_dsi has {model.size} values but {len(cts)} cell types")
    table = {"vector_sum": KNOWN_DSI_VECTOR_SUM, "classic": KNOWN_DSI_CLASSIC}
    if convention not in table:
        raise ValueError(f"convention must be one of {sorted(table)}")
    missing = [ct for ct in cts if ct not in table[convention]]
    if missing:
        raise KeyError(f"no DSI target for cell types {missing}")
    target = np.array([table[convention][ct] for ct in cts])
    ae = np.abs(model - target)
    r = float(_pearson_rows(model[None, :], target[None, :])[0])
    return DSICorrespondence(
        cell_types=cts,
        model_dsi=model,
        target_dsi=target,
        absolute_error=ae,
        mean_absolute_error=float(ae.mean()),
        pearson_r=r,
        convention=convention,
    )


def dsi_correlation_to_known(
    model_dsi: Mapping[str, float] | np.ndarray,
    cell_types: Optional[Sequence[str]] = None,
) -> float:
    """flyvis's `dsi_correlation_to_known`: r between model DSI and a BINARY target.

    Definition (flyvis/analysis/moving_bar_responses.py::dsi_correlation_to_known)
    ----------------------------------------------------------------------------
    Restrict to the 18 cell types with a known motion-tuning label:
    MOTION_TUNING_TYPES (8: T4a-d, T5a-d) with target 1, and
    NO_MOTION_TUNING_TYPES (10: Tm1, Tm2, Tm4, Tm9, CT1(Lo1), Mi1, Tm3, Mi4,
    Mi9, CT1(M10)) with target 0. Then

        rho = pearson_r over those 18 cell types ( model_dsi , binary_target ).

    In flyvis the model DSI is first max-aggregated over the intensity axis;
    do that yourself before calling (`dsi.max(axis=intensity)`).

    This is one of the FOUR quantities [Karaneen2026] plots as violins ("DSI"),
    and it is stock flyvis, so this is a close reimplementation rather than an
    approximation. The one thing you must supply the same way flyvis does is
    the DSI itself -- see the caveat in `direction_selectivity_index`.

    Note the target is BINARY, not graded: this measures whether the model gets
    the direction-selective/not-direction-selective ORDERING right, not whether
    it reproduces DSI magnitudes.

    Parameters
    ----------
    model_dsi : mapping cell type -> DSI, or an array aligned with `cell_types`.
    """
    if isinstance(model_dsi, _ABCMapping):
        values = dict(model_dsi)
    else:
        if cell_types is None:
            raise ValueError("cell_types is required when model_dsi is an array")
        arr = _asnumpy(model_dsi).ravel()
        if arr.size != len(cell_types):
            raise ValueError("model_dsi length must match cell_types")
        values = {ct: float(v) for ct, v in zip(cell_types, arr)}

    known = list(MOTION_TUNING_TYPES) + list(NO_MOTION_TUNING_TYPES)
    present = [ct for ct in known if ct in values]
    if len(present) < 3:
        raise ValueError(
            "need at least 3 of the 18 labelled cell types; got "
            f"{len(present)}: {present}"
        )
    x = _asnumpy([values[ct] for ct in present])
    y = np.array([1.0 if ct in MOTION_TUNING_TYPES else 0.0 for ct in present])
    return float(_pearson_rows(x[None, :], y[None, :])[0])


def preferred_direction_correspondence(
    theta_rad: np.ndarray,
    cell_types: Sequence[str] = T4T5_CELL_TYPES,
) -> Dict[str, float]:
    """Angular distance (deg) between a model's preferred directions and biology.

    Definition: for each cell type c,
        err_c = | ((deg(theta_c) - KNOWN_PREFERRED_DIRECTIONS_DEG[c] + 180) mod 360) - 180 |
    in [0, 180]. Tracks flyvis's `angular_distance_to_known` (which scores T4
    subtypes at intensity=1 and T5 subtypes at intensity=0).

    Returns a dict per cell type plus "mean_abs_error_deg".
    See the CONVENTION WARNING on KNOWN_PREFERRED_DIRECTIONS_DEG before
    comparing these numbers to any other paper's degree values.
    """
    theta = _asnumpy(theta_rad).ravel()
    cts = tuple(cell_types)
    if theta.size != len(cts):
        raise ValueError("theta_rad length must match cell_types")
    deg = np.rad2deg(theta) % 360.0
    out: Dict[str, float] = {}
    errs = []
    for ct, d in zip(cts, deg):
        if ct not in KNOWN_PREFERRED_DIRECTIONS_DEG:
            raise KeyError(f"no known preferred direction for {ct!r}")
        e = float(angular_distance_deg(d, KNOWN_PREFERRED_DIRECTIONS_DEG[ct]))
        out[ct] = e
        errs.append(e)
    out["mean_abs_error_deg"] = float(np.mean(errs))
    return out


def flash_response_index(
    on_peak: np.ndarray, off_peak: np.ndarray, *, nonnegative: bool = True
) -> np.ndarray:
    """Flash response index (FRI), flyvis definition.

    Definition (flyvis/analysis/flash_responses.py::flash_response_index)
    --------------------------------------------------------------------
        FRI = (on_peak - off_peak) / (on_peak + off_peak + 1e-16)

    where on_peak / off_peak are the maxima over the flash response window
    (`-dt <= t <= t_stim`) at stimulus radius 6, for intensity 1.0 and 0.0
    respectively. FRI is +1 for a purely ON cell, -1 for purely OFF.

    APPROXIMATION NOTE: flyvis additionally requires the flash stimulus to have
    `alternations == (0, 1, 0)`, selects `radius == 6`, and -- when
    `nonnegative=True`, the default -- shifts responses by
    `+|min over (frame, sample)|` BEFORE taking the peaks. That shift is applied
    to the full response time series, which this function does not receive, so
    it cannot be reproduced here; `nonnegative` only guards against a negative
    denominator. Compute the peaks with flyvis's own pipeline if you need
    published-comparable FRIs.
    """
    on = _asnumpy(on_peak)
    off = _asnumpy(off_peak)
    if nonnegative:
        shift = np.minimum(np.minimum(on, off), 0.0)
        on = on - shift
        off = off - shift
    return (on - off) / (on + off + _EPS_FLYVIS_FRI)


def fri_correlation_to_known(
    fri: Mapping[str, float], polarity: Mapping[str, int]
) -> float:
    """flyvis's `fri_correlation_to_known`: r between model FRI and ON/OFF polarity.

    Definition: Pearson correlation, over cell types with a known non-zero
    polarity, between the model FRI vector and the literature polarity vector
    g in {-1, +1}. This is the "FRI" violin in [Karaneen2026].

    WHY `polarity` IS A REQUIRED ARGUMENT, NOT A SHIPPED CONSTANT
    ------------------------------------------------------------
    flyvis's table (`flyvis/utils/groundtruth_utils.py::polarity`) has 66 cell
    types, 32 of them with non-zero polarity (R1-R8, L1-L5, C3, CT1(Lo1),
    CT1(M10), Mi1, Mi4, Mi9, T4a-d, T5a-d, Tm1, Tm2, Tm3, Tm4, Tm9), each with
    its own inline citation (e.g. "L5: 1  # Drews 2020, Matulis 2020"). We did
    NOT verify those 32 signs one-by-one against their cited sources, and
    fabricating them here would be exactly the kind of unverified constant this
    module is meant to avoid. Import the dict from flyvis, or transcribe it
    yourself from the source file, and pass it in. Cell types with polarity 0
    are dropped, matching flyvis.
    """
    types = [ct for ct in fri if ct in polarity and polarity[ct] != 0]
    if len(types) < 3:
        raise ValueError(
            f"need at least 3 cell types with non-zero polarity; got {len(types)}"
        )
    x = np.array([float(fri[ct]) for ct in types])
    y = np.array([float(polarity[ct]) for ct in types])
    return float(_pearson_rows(x[None, :], y[None, :])[0])


# =============================================================================
# 7. TWO-ARM VARIANCE INFERENCE (this is a SCALE comparison, not a mean one)
# =============================================================================

@dataclass(frozen=True)
class VarianceTestResult:
    statistic: float
    pvalue: float
    method: str
    var_a: float
    var_b: float
    log_var_ratio: float
    n_a: int
    n_b: int
    df: Optional[Tuple[float, float]] = None

    @property
    def var_ratio(self) -> float:
        return float(np.exp(self.log_var_ratio))

    def summary(self) -> str:
        return (
            f"{self.method}: stat={self.statistic:.4f} p={self.pvalue:.4g} "
            f"var_ratio={self.var_ratio:.3f} (log={self.log_var_ratio:+.3f}) "
            f"n=({self.n_a},{self.n_b})"
        )


def _bf_pvalues(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Brown-Forsythe over rows. A: (S, na), B: (S, nb) -> (W, p)."""
    za = np.abs(A - np.median(A, axis=1, keepdims=True))
    zb = np.abs(B - np.median(B, axis=1, keepdims=True))
    na, nb = A.shape[1], B.shape[1]
    N, k = na + nb, 2
    ma, mb = za.mean(axis=1), zb.mean(axis=1)
    grand = (na * ma + nb * mb) / N
    num = na * (ma - grand) ** 2 + nb * (mb - grand) ** 2
    den = ((za - ma[:, None]) ** 2).sum(axis=1) + ((zb - mb[:, None]) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = ((N - k) / (k - 1)) * num / den
    p = stats.f.sf(W, k - 1, N - k)
    bad = ~np.isfinite(W)
    # den == 0 means all deviations identical within both arms: no evidence.
    W = np.where(bad, 0.0, W)
    p = np.where(bad, 1.0, p)
    return W, p


def brown_forsythe_variance_test(a: np.ndarray, b: np.ndarray) -> VarianceTestResult:
    """SECONDARY: Brown-Forsythe test for equality of SCALE between two arms.

    DEMOTED FROM PRIMARY IN ROUND 2 (design decision D3 / issue science-3).
    Brown-Forsythe is a correct and well-calibrated test FOR N INDEPENDENT
    SCALARS PER ARM -- the tables below still hold, and the smoke test still
    reproduces them. It is NOT calibrated for the endpoint this study reports,
    because the per-seed deviations fed to it are distances to a centre
    estimated from the same seeds, hence dependent; with `center="medoid"` one
    of them is exactly zero and dominates the denominator. Measured
    consequences (smoke test section [8], this machine, this session): size far
    below nominal and near-zero power at a true 3x dispersion ratio on
    96-dimensional fingerprints. The primary test is `permdisp`.
    Use this function for a genuinely scalar per-seed endpoint -- e.g. a single
    correlation-to-ground-truth per model (M3), or the per-seed task error --
    where the independence assumption actually holds.

    Definition ([BrownForsythe1974]; Levene's test centred on the MEDIAN)
    --------------------------------------------------------------------
    For arms with observations y_{1,1..n1} and y_{2,1..n2}, form the absolute
    deviations from each arm's own median

        z_{ij} = | y_{ij} - median_i |,

    then compute the one-way ANOVA F statistic on the z:

        W = ((N - k) / (k - 1)) * sum_i n_i (zbar_i - zbar)^2
                               / sum_i sum_j (z_ij - zbar_i)^2

    with k = 2 arms, N = n1 + n2, referred to F(k-1, N-k). Identical to
    `scipy.stats.levene(a, b, center='median')` (the smoke test asserts this).

    WHY NOT THE F-TEST OF VARIANCES
    -------------------------------
    The textbook variance-ratio test F = s_A^2 / s_B^2 ~ F(n-1, n-1) is the
    wrong choice here. Its Type-I error is driven by KURTOSIS, not variance, and
    does not improve with n. Simulated rejection rates at nominal alpha = 0.05,
    n = 50 per arm, 20,000 reps (design research for this study):

        distribution        F-test    Brown-Forsythe
        normal               0.048        0.044
        t_5 (heavy-tailed)   0.199 BAD    0.047
        lognormal (skewed)   0.329 BAD    0.050
        beta(5, 1.5)         0.068        0.046

    Under t_5 the F-test got WORSE with n (0.128 at n=10 -> 0.226 at n=150). Our
    metrics are bounded and skewed (correlations to ground truth, angular
    deviations in [0, pi], distances >= 0), so the F-test is unusable. See
    `variance_f_test`, which exists only to demonstrate this.

    WHY NOT A t-TEST / MANN-WHITNEY
    -------------------------------
    Those compare LOCATION. The study's claim is about SCALE (mechanism variance
    across seeds). "Weaker" and "more variable" are two different claims; test
    them separately and report both.

    ALWAYS report the effect size alongside p: `log_variance_ratio_ci`.
    """
    a = _check_1d(a, "a")
    b = _check_1d(b, "b")
    W, p = _bf_pvalues(a[None, :], b[None, :])
    return VarianceTestResult(
        statistic=float(W[0]),
        pvalue=float(p[0]),
        method="Brown-Forsythe (Levene, median-centred)",
        var_a=float(np.var(a, ddof=1)),
        var_b=float(np.var(b, ddof=1)),
        log_var_ratio=log_variance_ratio(a, b),
        n_a=a.size,
        n_b=b.size,
        df=(1.0, float(a.size + b.size - 2)),
    )


def log_variance_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """log( var(a, ddof=1) / var(b, ddof=1) ) -- the effect size to report."""
    a = _check_1d(a, "a")
    b = _check_1d(b, "b")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    if va <= 0 or vb <= 0:
        return float("nan")
    return float(np.log(va / vb))


def log_variance_ratio_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    method: str = "bca",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Bootstrap CI for log(var_A / var_B) -- the interpretable effect size.

    Definition
    ----------
    theta = log(s_A^2 / s_B^2). Resample seeds WITH REPLACEMENT within each arm
    (seeds are the independent unit), recompute theta, and form either

      method="percentile": [ theta*_(alpha/2) , theta*_(1-alpha/2) ]
      method="bca" (default): bias-corrected and accelerated
          [Efron & Tibshirani, An Introduction to the Bootstrap (1993), ch. 14],
          with
              z0 = Phi^-1( #{theta* < theta_hat} / B )
              a  = sum_i (thetabar_(.) - theta_(i))^3
                   / ( 6 * ( sum_i (thetabar_(.) - theta_(i))^2 )^{3/2} )
          from delete-one jackknife replicates over the pooled n_A + n_B seeds,
          and endpoints
              alpha1 = Phi( z0 + (z0 + z_{alpha/2})   / (1 - a (z0 + z_{alpha/2})) )
              alpha2 = Phi( z0 + (z0 + z_{1-alpha/2}) / (1 - a (z0 + z_{1-alpha/2})) ).

    A CI excluding 0 is equivalent to rejecting equal variances; report the
    ratio and its interval, never p alone. Returns a dict with
    `log_var_ratio`, `var_ratio`, `ci_low`, `ci_high` (on the log scale),
    `ratio_ci_low`, `ratio_ci_high` (exponentiated), and the BCa nuisance
    parameters.
    """
    a = _check_1d(a, "a")
    b = _check_1d(b, "b")
    rng = np.random.default_rng() if rng is None else rng
    na, nb = a.size, b.size
    theta_hat = log_variance_ratio(a, b)

    ia = rng.integers(0, na, size=(n_boot, na))
    ib = rng.integers(0, nb, size=(n_boot, nb))
    va = a[ia].var(axis=1, ddof=1)
    vb = b[ib].var(axis=1, ddof=1)
    ok = (va > 0) & (vb > 0)
    boot = np.log(va[ok] / vb[ok])
    if boot.size < 100:
        raise RuntimeError("bootstrap degenerated (too many zero-variance resamples)")

    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    if method == "percentile":
        lo, hi = np.percentile(boot, [100 * lo_q, 100 * hi_q])
        z0 = acc = float("nan")
    elif method == "bca":
        prop = float(np.mean(boot < theta_hat))
        prop = min(max(prop, 1.0 / (boot.size + 1)), 1.0 - 1.0 / (boot.size + 1))
        z0 = float(stats.norm.ppf(prop))
        jack = []
        for i in range(na):
            jack.append(log_variance_ratio(np.delete(a, i), b))
        for j in range(nb):
            jack.append(log_variance_ratio(a, np.delete(b, j)))
        jack = np.asarray(jack, dtype=float)
        jack = jack[np.isfinite(jack)]
        dev = jack.mean() - jack
        denom = 6.0 * (float((dev ** 2).sum()) ** 1.5)
        acc = float((dev ** 3).sum() / denom) if denom > 0 else 0.0
        z_lo, z_hi = stats.norm.ppf(lo_q), stats.norm.ppf(hi_q)
        a1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - acc * (z0 + z_lo)))
        a2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - acc * (z0 + z_hi)))
        lo, hi = np.percentile(boot, [100 * a1, 100 * a2])
    else:
        raise ValueError("method must be 'bca' or 'percentile'")

    return {
        "log_var_ratio": theta_hat,
        "var_ratio": float(np.exp(theta_hat)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ratio_ci_low": float(np.exp(lo)),
        "ratio_ci_high": float(np.exp(hi)),
        "alpha": float(alpha),
        "n_boot": int(boot.size),
        "bca_z0": float(z0),
        "bca_acceleration": float(acc),
        "method": method,
    }


def permutation_variance_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_perm: int = 9999,
    rng: Optional[np.random.Generator] = None,
) -> VarianceTestResult:
    """CONFIRMATORY: exact permutation test on |log(var_A / var_B)|.

    Definition
    ----------
    Median-centre EACH ARM first (a' = a - median(a), b' = b - median(b)) so
    that under H0 the two arms differ only in SCALE -- without this, a location
    shift leaks into the null distribution and inflates the rejection rate.
    Then with T = |log(var(a') / var(b'))|, pool a' and b', draw `n_perm` random
    splits into groups of size n_A and n_B, and

        p = (#{T_perm >= T_obs} + 1) / (n_perm + 1).

    Distribution-free given exchangeability of the centred values; the natural
    companion to Brown-Forsythe when the metric is bounded or heavy-tailed.
    """
    a = _check_1d(a, "a")
    b = _check_1d(b, "b")
    rng = np.random.default_rng() if rng is None else rng
    ac = a - np.median(a)
    bc = b - np.median(b)
    t_obs = abs(float(np.log(np.var(ac, ddof=1) / np.var(bc, ddof=1))))
    pool = np.concatenate([ac, bc])
    na = ac.size
    perms = rng.permuted(np.tile(pool, (n_perm, 1)), axis=1)
    va = perms[:, :na].var(axis=1, ddof=1)
    vb = perms[:, na:].var(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_perm = np.abs(np.log(va / vb))
    t_perm = t_perm[np.isfinite(t_perm)]
    p = (int(np.sum(t_perm >= t_obs)) + 1) / (t_perm.size + 1)
    return VarianceTestResult(
        statistic=t_obs,
        pvalue=float(p),
        method=f"permutation on |log var ratio| ({t_perm.size} perms, median-centred)",
        var_a=float(np.var(a, ddof=1)),
        var_b=float(np.var(b, ddof=1)),
        log_var_ratio=log_variance_ratio(a, b),
        n_a=a.size,
        n_b=b.size,
    )


def variance_f_test(a: np.ndarray, b: np.ndarray) -> VarianceTestResult:
    """DO NOT USE FOR INFERENCE. Classical F test of variances, F = s_A^2/s_B^2.

    Definition: F = var(a, ddof=1) / var(b, ddof=1) ~ F(n_a - 1, n_b - 1) under
    normality, two-sided p = 2 * min(cdf, sf).

    This function exists so the smoke test can DEMONSTRATE the failure, and so
    that anyone reaching for the textbook test finds this docstring first. Its
    Type-I error under a t_5 metric is ~0.20 at nominal 0.05, and ~0.33 under a
    lognormal one -- see `brown_forsythe_variance_test` for the table. The
    failure is driven by kurtosis and does not shrink with n. Use
    `brown_forsythe_variance_test` (primary) and `permutation_variance_test`
    (confirmatory) instead.
    """
    a = _check_1d(a, "a")
    b = _check_1d(b, "b")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    f = float(va / vb)
    dfa, dfb = a.size - 1, b.size - 1
    p = 2.0 * min(stats.f.cdf(f, dfa, dfb), stats.f.sf(f, dfa, dfb))
    return VarianceTestResult(
        statistic=f,
        pvalue=float(min(p, 1.0)),
        method="classical F test of variances (NOT ROBUST -- do not use)",
        var_a=float(va),
        var_b=float(vb),
        log_var_ratio=float(np.log(f)),
        n_a=a.size,
        n_b=b.size,
        df=(float(dfa), float(dfb)),
    )


# ---------------------------------------------------------------------------
# Power / sample size
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PowerResult:
    n_per_arm: Optional[int]
    achieved_power: float
    target_power: float
    var_ratio: float
    alpha: float
    dist: str
    test: str
    n_sims: int
    note: str = ""

    def summary(self) -> str:
        n = "not reached" if self.n_per_arm is None else str(self.n_per_arm)
        return (
            f"var_ratio={self.var_ratio:g} dist={self.dist} test={self.test}: "
            f"n_per_arm={n} (power={self.achieved_power:.3f}, "
            f"target={self.target_power:g}, alpha={self.alpha:g}, "
            f"n_sims={self.n_sims})"
        )


def _unit_variance_variates(
    rng: np.random.Generator, shape: Tuple[int, int], dist: str
) -> np.ndarray:
    """Draw mean-0, variance-1 variates from a named distribution."""
    if dist == "normal":
        return rng.standard_normal(shape)
    if dist == "t5":
        # Var(t_5) = 5/3; rescale so the "variance ratio" argument means what it says.
        return rng.standard_t(5, size=shape) / math.sqrt(5.0 / 3.0)
    if dist == "lognormal":
        x = rng.lognormal(mean=0.0, sigma=1.0, size=shape)
        m = math.exp(0.5)
        sd = math.sqrt((math.e - 1.0) * math.e)
        return (x - m) / sd
    if dist == "uniform":
        return (rng.random(shape) - 0.5) * math.sqrt(12.0)
    raise ValueError(f"unknown dist {dist!r}; use normal|t5|lognormal|uniform")


def _power_from_variates(
    base_a: np.ndarray, base_b: np.ndarray, n: int, var_ratio: float, alpha: float, test: str
) -> float:
    """Rejection rate at sample size n, using pre-drawn common random numbers."""
    A = base_a[:, :n] * math.sqrt(var_ratio)
    B = base_b[:, :n]
    if test == "brown_forsythe":
        _, p = _bf_pvalues(A, B)
    elif test == "f":
        va = A.var(axis=1, ddof=1)
        vb = B.var(axis=1, ddof=1)
        f = va / vb
        p = 2.0 * np.minimum(stats.f.cdf(f, n - 1, n - 1), stats.f.sf(f, n - 1, n - 1))
    else:
        raise ValueError("test must be 'brown_forsythe' or 'f'")
    return float(np.mean(p < alpha))


def variance_test_power(
    n_per_arm: int,
    var_ratio: float,
    *,
    alpha: float = 0.05,
    dist: str = "normal",
    test: str = "brown_forsythe",
    n_sims: int = 4000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Monte-Carlo power to detect a given VARIANCE RATIO between two arms.

    Definition
    ----------
    Simulate `n_sims` two-arm experiments with n = `n_per_arm` seeds per arm,
    arm A drawn with variance `var_ratio` times arm B's (both arms from `dist`,
    standardized to unit variance then scaled), run `test` at level `alpha`
    two-sided, and return the rejection rate. `var_ratio` is the TRUE variance
    ratio; the corresponding SD ratio is sqrt(var_ratio).

    Assumes seeds are i.i.d. within an arm, the two arms differ ONLY in scale,
    and the reported metric is a single PRE-REGISTERED scalar. Testing all eight
    T4/T5 subtypes and Holm-correcting raises the required N by roughly 25-40%,
    which this helper does NOT model.

    Reference points (design research, 20k reps): at n = 50 per arm with
    Brown-Forsythe and a normal metric, power is ~0.24 (ratio 1.5), ~0.58
    (2.0), ~0.81 (2.5), ~0.93 (3.0), ~0.99 (4.0). Heavy tails cost ~0.15
    of power at ratio 2-3.
    """
    if n_per_arm < 3:
        raise ValueError("n_per_arm must be >= 3")
    rng = np.random.default_rng() if rng is None else rng
    base_a = _unit_variance_variates(rng, (n_sims, n_per_arm), dist)
    base_b = _unit_variance_variates(rng, (n_sims, n_per_arm), dist)
    return _power_from_variates(base_a, base_b, n_per_arm, var_ratio, alpha, test)


def n_per_arm_for_variance_ratio(
    var_ratio: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    dist: str = "normal",
    test: str = "brown_forsythe",
    n_sims: int = 2000,
    n_min: int = 5,
    n_max: int = 600,
    rng: Optional[np.random.Generator] = None,
) -> PowerResult:
    """POWER HELPER: seeds per arm needed to detect a given variance ratio.

    Each "seed" is one full training run, so this number is the study's budget.

    Method
    ------
    Monte-Carlo bisection on n. Common random numbers are drawn ONCE at size
    (n_sims, n_max) and sliced, which makes the simulated power curve monotone
    in n and the bisection stable. Returns the smallest n in [n_min, n_max] whose
    simulated power reaches `power`, or n_per_arm=None if `n_max` does not suffice.

    Monte-Carlo error: with `n_sims` reps the power estimate has SE ~
    sqrt(p(1-p)/n_sims) (~0.9% at n_sims=2000, p=0.8), so expect the returned n
    to wobble by roughly +-15% between runs. Increase `n_sims` (and pass a fixed
    `rng`) before quoting a number in a paper.

    Sanity anchors for Brown-Forsythe / normal / 80% power / alpha 0.05 -- see
    N_PER_ARM_REFERENCE_BF_NORMAL: ratio 1.5 -> ~225, 2.0 -> ~80, 2.5 -> ~49,
    3.0 -> ~36, 4.0 -> ~24. A t_5 metric needs roughly 1.4x those.

    DESIGN RULE THAT FALLS OUT OF THIS TABLE: a variance comparison needs about
    4x the N of the corresponding mean comparison. N = 50 per arm (one flyvis
    ensemble) detects only ratios >= 2.5-3 -- a screen, not a subtle claim.
    N < 30 per arm cannot support any variance claim below a 3-4x ratio.
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if var_ratio <= 0:
        raise ValueError("var_ratio must be > 0")
    rng = np.random.default_rng() if rng is None else rng
    base_a = _unit_variance_variates(rng, (n_sims, n_max), dist)
    base_b = _unit_variance_variates(rng, (n_sims, n_max), dist)

    def power_at(n: int) -> float:
        return _power_from_variates(base_a, base_b, n, var_ratio, alpha, test)

    p_max = power_at(n_max)
    if p_max < power:
        return PowerResult(
            n_per_arm=None, achieved_power=p_max, target_power=power,
            var_ratio=var_ratio, alpha=alpha, dist=dist, test=test, n_sims=n_sims,
            note=f"power at n_max={n_max} is only {p_max:.3f}; raise n_max",
        )
    lo, hi = n_min, n_max
    if power_at(lo) >= power:
        return PowerResult(
            n_per_arm=lo, achieved_power=power_at(lo), target_power=power,
            var_ratio=var_ratio, alpha=alpha, dist=dist, test=test, n_sims=n_sims,
            note=f"target power already met at n_min={n_min}",
        )
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if power_at(mid) >= power:
            hi = mid
        else:
            lo = mid
    return PowerResult(
        n_per_arm=hi, achieved_power=power_at(hi), target_power=power,
        var_ratio=var_ratio, alpha=alpha, dist=dist, test=test, n_sims=n_sims,
    )


# ---------------------------------------------------------------------------
# END-TO-END POWER SIMULATION OF THE PRIMARY PIPELINE  (design decision D3)
# ---------------------------------------------------------------------------
#
# The scalar power helpers above answer "how many i.i.d. normal numbers per arm
# do I need". That is NOT the study's endpoint. The endpoint is
#
#     synthetic fingerprints -> L1 normalization -> Euclidean distance matrix
#     -> PERMDISP -> p
#
# and the only honest way to size it is to run exactly that, which is what the
# functions below do. Two consequences, both of which must be carried into any
# budget:
#
# (1) THE REQUIRED N DEPENDS ON THE COVARIANCE SPECTRUM OF THE ACROSS-SEED
#     VARIATION, not just on the variance ratio. PERMDISP's statistic is an
#     ANOVA on z_s = ||m_s - centroid||. If the across-seed variation spreads
#     over many independent directions, each z_s averages over them: mean(z)
#     grows like sqrt(p_eff) while sd(z) stays O(sigma), so the standardized
#     between-arm effect grows like sqrt(p_eff) and N falls fast. If the
#     variation is concentrated in one or two directions (p_eff ~ 1-2), z
#     behaves like a scalar magnitude and the required N is much larger. The
#     spectrum is an EMPIRICAL property of the released ensemble: estimate
#     p_eff = (sum lambda)^2 / sum lambda^2 from the 50 released models in
#     Stage 0 (free, no training) and quote the corresponding row. Pre-register
#     the conservative row you are willing to fund.
# (2) The variance ratio must be the REALIZED ratio of `ensemble_total_variance`
#     between arms, not the generator's nominal sigma ratio: the L1
#     normalization is nonlinear. `calibrate_sigma_for_variance_ratio` solves
#     for the generator scale that produces the requested realized ratio, and
#     the simulation reports the realized value it achieved.


@dataclass(frozen=True)
class PipelineSpec:
    """Generator for synthetic mechanism fingerprints (the simulation's H1).

    n_source_types x n_conditions is the fingerprint dimension p. The default
    12 x 2 = 24 is the scale of a T4c current decomposition; set
    n_source_types=8, n_conditions=12 to simulate the 96-dimensional SECONDARY
    response fingerprint instead.

    effective_rank is the participation ratio p_eff = (sum l)^2 / sum l^2 of the
    across-seed covariance spectrum (l_k proportional to k^-a, with a solved to
    hit the requested p_eff). p_eff = p means isotropic variation; p_eff -> 1
    means all seeds differ along one direction.

    dist selects the latent distribution: "normal", "t5" (heavy-tailed) or
    "lognormal" (skewed), each standardized to unit variance.

    Model: m_raw = base * exp(sigma * z), z ~ N(0, Q diag(l) Q^T) with Q a fixed
    orthonormal basis (basis_seed), then the ACTUAL pipeline normalization is
    applied. Multiplicative perturbation is used because it preserves the sign
    of each presynaptic current, which a mechanism fingerprint must keep.
    """

    n_source_types: int = 12
    n_conditions: int = 2
    effective_rank: float = 4.0
    dist: str = "normal"
    normalize: str = "l1"
    metric: str = "euclidean"
    center: str = "centroid"
    basis_seed: int = 20260915

    @property
    def n_features(self) -> int:
        return self.n_source_types * self.n_conditions


def _spectrum_for_effective_rank(p: int, p_eff: float) -> np.ndarray:
    """Eigenvalues l_k ~ k^-a, normalized to sum 1, with participation ratio p_eff."""
    if not (1.0 <= p_eff <= p):
        raise ValueError(f"effective_rank must be in [1, {p}], got {p_eff}")
    k = np.arange(1, p + 1, dtype=float)

    def pr(a: float) -> float:
        l = k ** (-a)
        l = l / l.sum()
        return float(1.0 / (l ** 2).sum())

    lo, hi = 0.0, 60.0
    if pr(hi) > p_eff:  # cannot concentrate further
        a = hi
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if pr(mid) > p_eff:
                lo = mid
            else:
                hi = mid
        a = 0.5 * (lo + hi)
    l = k ** (-a)
    return l / l.sum()


def _pipeline_base(spec: PipelineSpec) -> np.ndarray:
    """Deterministic base current decomposition, shape (n_source_types, n_conditions).

    A stylized T4c-like profile: magnitudes decaying across presynaptic types,
    every third type inhibitory (negative), and the preferred direction carrying
    more drive than the null direction. Nothing here is a measured quantity --
    it only fixes a realistic sign pattern and dynamic range for the simulation.
    """
    s = np.arange(spec.n_source_types, dtype=float)
    mag = 1.0 / (1.0 + 0.6 * s)
    sign = np.where((np.arange(spec.n_source_types) % 3) == 2, -1.0, 1.0)
    cond = np.linspace(1.0, 0.45, spec.n_conditions)
    return (mag * sign)[:, None] * cond[None, :]


def _pipeline_latents(
    spec: PipelineSpec, rng: np.random.Generator, shape: Tuple[int, ...]
) -> np.ndarray:
    """Draw standardized latent fingerprint perturbations of shape (*shape, p)."""
    p = spec.n_features
    lam = _spectrum_for_effective_rank(p, spec.effective_rank)
    basis_rng = np.random.default_rng(spec.basis_seed)
    Q, _ = np.linalg.qr(basis_rng.standard_normal((p, p)))
    e = _unit_variance_variates(rng, (int(np.prod(shape)), p), spec.dist)
    # scale so that total latent variance is 1 regardless of the spectrum
    z = (e * np.sqrt(lam * p)) @ Q.T
    return z.reshape(*shape, p)


def _fingerprints_from_latents(
    spec: PipelineSpec, latents: np.ndarray, sigma: float
) -> np.ndarray:
    """Apply the ACTUAL pipeline: perturb the base, then normalize."""
    base = _pipeline_base(spec).reshape(-1)
    raw = base * np.exp(sigma * latents)
    return _normalize_last_axis(raw, spec.normalize)


def synthetic_current_ensemble(
    rng: np.random.Generator,
    n_models: int,
    *,
    spec: Optional[PipelineSpec] = None,
    sigma: float = 0.35,
    arm: str = "",
    source_types: Optional[Sequence[str]] = None,
) -> CurrentEnsemble:
    """Synthesize a `CurrentEnsemble` for testing the pipeline end to end.

    Seed-to-seed variation is multiplicative on each presynaptic current
    (`sigma` in log units, spread over the covariance spectrum given by `spec`),
    so signs are preserved and the L1 normalization has something real to
    remove. This is the generator the power simulation uses; it is NOT a model
    of the fly.
    """
    spec = spec or PipelineSpec()
    z = _pipeline_latents(spec, rng, (n_models,))
    base = _pipeline_base(spec)
    raw = base.reshape(-1) * np.exp(sigma * z)
    contributions = raw.reshape(n_models, spec.n_source_types, spec.n_conditions)
    sts = (
        tuple(source_types)
        if source_types is not None
        else tuple(f"pre{i:02d}" for i in range(spec.n_source_types))
    )
    conds = CURRENT_CONDITIONS if spec.n_conditions == 2 else tuple(
        f"c{i}" for i in range(spec.n_conditions)
    )
    return CurrentEnsemble(
        contributions=contributions, source_types=sts, conditions=conds, arm=arm
    )


def calibrate_sigma_for_variance_ratio(
    var_ratio: float,
    *,
    spec: Optional[PipelineSpec] = None,
    sigma_ref: float = 0.35,
    n_ref: int = 4000,
    seed: int = 20260915,
    tol: float = 1e-3,
) -> Dict[str, float]:
    """Solve for the generator scale that yields a REALIZED variance ratio.

    The pipeline is nonlinear (multiplicative perturbation then L1
    normalization), so doubling sigma does not multiply
    `ensemble_total_variance` by exactly 4. This routine bisects on sigma_b so
    that V(sigma_b) / V(sigma_ref) equals `var_ratio` on a large reference
    sample, with common random numbers across evaluations (making V(sigma)
    smooth and monotone). Deterministic given `seed`.

    Returns `sigma_a`, `sigma_b`, `V_a`, `V_b`, `realized_ratio`, `n_ref`.
    """
    spec = spec or PipelineSpec()
    if var_ratio < 1.0:
        raise ValueError("var_ratio must be >= 1 (arm B is the more dispersed arm)")
    rng = np.random.default_rng(seed)
    latents = _pipeline_latents(spec, rng, (n_ref,))

    def V(sigma: float) -> float:
        return ensemble_total_variance(_fingerprints_from_latents(spec, latents, sigma))

    v_a = V(sigma_ref)
    target = var_ratio * v_a
    lo, hi = sigma_ref, sigma_ref
    for _ in range(60):
        if V(hi) >= target:
            break
        hi *= 1.5
    else:  # pragma: no cover - would need an absurd ratio
        raise RuntimeError("could not bracket the target variance ratio")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if V(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * sigma_ref:
            break
    sigma_b = 0.5 * (lo + hi)
    v_b = V(sigma_b)
    return {
        "sigma_a": float(sigma_ref),
        "sigma_b": float(sigma_b),
        "V_a": float(v_a),
        "V_b": float(v_b),
        "realized_ratio": float(v_b / v_a),
        "n_ref": float(n_ref),
    }


def simulate_permdisp_power(
    n_per_arm: int,
    var_ratio: float,
    *,
    spec: Optional[PipelineSpec] = None,
    alpha: float = 0.05,
    n_sims: int = 400,
    n_perm: int = 199,
    sigma_ref: float = 0.35,
    rng: Optional[np.random.Generator] = None,
    calibration: Optional[Mapping[str, float]] = None,
    also_brown_forsythe: bool = False,
) -> Dict[str, float]:
    """END-TO-END power (or size, at var_ratio=1) of the PRIMARY pipeline.

    Runs the whole thing, `n_sims` times:
        generate n_per_arm fingerprints per arm with a calibrated realized
        variance ratio -> `fingerprint_distance_matrix` (Euclidean) ->
        `permdisp` (centroid, label permutation) -> reject if p < alpha.

    At `var_ratio=1.0` the returned `power` IS the false-positive rate, which is
    how the smoke test verifies that the primary test holds its nominal size on
    null data.

    `also_brown_forsythe=True` additionally runs the demoted scalar route
    (Brown-Forsythe on medoid deviations) on the SAME simulated data, which is
    how the corrected power table documents the miscalibration side by side.

    Monte-Carlo error on the returned rate is sqrt(r(1-r)/n_sims) (2.5% at
    n_sims=400, r=0.5). `n_perm=199` makes the achievable level exactly
    10/200 = 0.05; raise it for a finer p-value, not for validity.

    Returns a dict: `power`, `alpha`, `n_per_arm`, `var_ratio_requested`,
    `var_ratio_realized`, `n_sims`, `n_perm`, `n_features`, `effective_rank`,
    `dist`, and (optionally) `power_bf_medoid`, `power_bf_mean_pairwise`.
    """
    spec = spec or PipelineSpec()
    if n_per_arm < 3:
        raise ValueError("n_per_arm must be >= 3")
    rng = np.random.default_rng() if rng is None else rng
    cal = calibration or calibrate_sigma_for_variance_ratio(
        var_ratio, spec=spec, sigma_ref=sigma_ref
    )
    sigma_a, sigma_b = float(cal["sigma_a"]), float(cal["sigma_b"])

    groups = np.array(["A"] * n_per_arm + ["B"] * n_per_arm)
    rejects = 0
    rej_bf_medoid = 0
    rej_bf_mean = 0
    for _ in range(n_sims):
        za = _pipeline_latents(spec, rng, (n_per_arm,))
        zb = _pipeline_latents(spec, rng, (n_per_arm,))
        fa = _fingerprints_from_latents(spec, za, sigma_a)
        fb = _fingerprints_from_latents(spec, zb, sigma_b)
        M = np.concatenate([fa, fb], axis=0)
        D = fingerprint_distance_matrix(M, metric=spec.metric)
        _, pval, _ = _permdisp_centroid_fast(D, groups, n_perm, rng)
        if pval < alpha:
            rejects += 1
        if also_brown_forsythe:
            da = deviations_from_center(D[:n_per_arm, :n_per_arm], center="medoid")
            db = deviations_from_center(D[n_per_arm:, n_per_arm:], center="medoid")
            if brown_forsythe_variance_test(db, da).pvalue < alpha:
                rej_bf_medoid += 1
            ma = deviations_from_center(D[:n_per_arm, :n_per_arm], center="mean_pairwise")
            mb = deviations_from_center(D[n_per_arm:, n_per_arm:], center="mean_pairwise")
            if brown_forsythe_variance_test(mb, ma).pvalue < alpha:
                rej_bf_mean += 1

    out = {
        "power": rejects / n_sims,
        "alpha": float(alpha),
        "n_per_arm": float(n_per_arm),
        "var_ratio_requested": float(var_ratio),
        "var_ratio_realized": float(cal["realized_ratio"]),
        "n_sims": float(n_sims),
        "n_perm": float(n_perm),
        "n_features": float(spec.n_features),
        "effective_rank": float(spec.effective_rank),
        "dist": spec.dist,
    }
    if also_brown_forsythe:
        out["power_bf_medoid"] = rej_bf_medoid / n_sims
        out["power_bf_mean_pairwise"] = rej_bf_mean / n_sims
    return out


def n_per_arm_for_permdisp(
    var_ratio: float,
    *,
    spec: Optional[PipelineSpec] = None,
    power: float = 0.8,
    alpha: float = 0.05,
    n_sims: int = 400,
    n_perm: int = 199,
    sigma_ref: float = 0.35,
    n_grid: Sequence[int] = (6, 8, 10, 12, 16, 20, 24, 30, 36, 48, 64, 80, 100, 125, 150),
    rng: Optional[np.random.Generator] = None,
) -> PowerResult:
    """SEEDS PER ARM for the PRIMARY endpoint, by simulating the actual pipeline.

    Walks `n_grid` upward and returns the first n whose simulated PERMDISP power
    reaches `power`. A grid rather than a bisection because the simulated power
    curve is noisy (independent draws at each n), so a bisection on a
    non-monotone estimate can terminate in the wrong place; the grid also lets
    the caller see the whole curve in `note`.

    THIS -- not `n_per_arm_for_variance_ratio` -- is the number that sizes the
    study. Quote it together with the `effective_rank` it assumes, because that
    assumption moves the answer by a factor of several (see the section comment
    above and `PIPELINE_POWER_TABLE_PERMDISP`).
    """
    spec = spec or PipelineSpec()
    rng = np.random.default_rng() if rng is None else rng
    cal = calibrate_sigma_for_variance_ratio(var_ratio, spec=spec, sigma_ref=sigma_ref)
    curve = []
    for n in n_grid:
        res = simulate_permdisp_power(
            n, var_ratio, spec=spec, alpha=alpha, n_sims=n_sims, n_perm=n_perm,
            sigma_ref=sigma_ref, rng=rng, calibration=cal,
        )
        curve.append((int(n), float(res["power"])))
        if res["power"] >= power:
            return PowerResult(
                n_per_arm=int(n),
                achieved_power=float(res["power"]),
                target_power=power,
                var_ratio=float(cal["realized_ratio"]),
                alpha=alpha,
                dist=spec.dist,
                test=f"PERMDISP(centroid) end-to-end, p={spec.n_features}, "
                     f"p_eff={spec.effective_rank:g}",
                n_sims=n_sims,
                note="power curve: " + ", ".join(f"{n}:{p:.2f}" for n, p in curve),
            )
    return PowerResult(
        n_per_arm=None,
        achieved_power=curve[-1][1],
        target_power=power,
        var_ratio=float(cal["realized_ratio"]),
        alpha=alpha,
        dist=spec.dist,
        test=f"PERMDISP(centroid) end-to-end, p={spec.n_features}, "
             f"p_eff={spec.effective_rank:g}",
        n_sims=n_sims,
        note="power curve: " + ", ".join(f"{n}:{p:.2f}" for n, p in curve)
             + f"; n_grid max {n_grid[-1]} insufficient",
    )


# =============================================================================
# 8. SMOKE TEST  (synthetic data, CPU)
# =============================================================================

def _synthetic_arm(
    rng: np.random.Generator,
    n_models: int,
    *,
    shape_noise: float,
    angle_jitter_deg: float,
    gain_cv: float = 0.15,
    arm: str = "",
) -> TuningEnsemble:
    """Synthesize an ensemble of 'trained models' around the measured T4/T5 curves.

    Each model = literature curve, circularly shifted by a per-seed angular
    jitter (this is what makes preferred directions disperse), multiplicatively
    gain-perturbed (which the L1 normalization is designed to remove), plus
    additive shape noise (which it is not). Larger `shape_noise` /
    `angle_jitter_deg` = a less reproducible arm.
    """
    angles = KNOWN_TUNING_ANGLES_DEG
    base = known_tuning_curves(T4T5_CELL_TYPES, angles)  # (8, 12)
    n_ct, n_ang = base.shape
    curves = np.empty((n_models, n_ct, n_ang))
    for s in range(n_models):
        jitter = rng.normal(0.0, angle_jitter_deg, size=n_ct)
        probe = angles[None, :] - jitter[:, None]
        # resample each cell type's periodic spline at the jittered angles
        shifted = np.stack([
            CubicSpline(
                np.append(angles, 360.0),
                np.append(KNOWN_TUNING_CURVES[ct], KNOWN_TUNING_CURVES[ct][0]),
                bc_type="periodic",
            )(np.mod(probe[i], 360.0))
            for i, ct in enumerate(T4T5_CELL_TYPES)
        ])
        gain = np.exp(rng.normal(0.0, gain_cv, size=(n_ct, 1)))
        curves[s] = np.clip(
            shifted * gain + rng.normal(0.0, shape_noise, size=(n_ct, n_ang)), 0.0, None
        )
    return TuningEnsemble(curves, T4T5_CELL_TYPES, angles, arm=arm)


class _FakeCurrentView:
    """Minimal stand-in for flyvis's `MovingEdgeCurrentView` (smoke test only).

    It mimics exactly two things, both read from flyvis v1.2.0 source
    (flyvis/analysis/moving_edge_currents.py @ 92b3845) on 2026-09-15:

      * `view[source_type][:]` -> the current trace array, which is how flyvis's
        own `get_summed_traces` reads a per-source-type trace;
      * `get_temporal_contributions(...)` -> (cv_pd, cv_nd, source_types,
        color_source_types), where the third element is CONTRIBUTION-SORTED and
        possibly truncated.

    The fake deliberately returns a WRONG third element (sorted by magnitude and
    truncated) so the smoke test can prove `from_flyvis_current_view` ignores it
    and indexes by the pre-registered names instead. It also asserts that the
    adapter passes `hide_source_types=None`, never "auto".
    """

    class _Leaf:
        def __init__(self, arr: np.ndarray) -> None:
            self._a = arr

        def __getitem__(self, key):
            if key == slice(None):
                return self._a
            raise KeyError(key)

    def __init__(self, traces_pd, traces_nd):
        self._pd = traces_pd
        self._nd = traces_nd
        self._which = "pd"
        self.calls = []

    def _select(self):
        return self._pd if self._which == "pd" else self._nd

    def __getitem__(self, key):
        if isinstance(key, str):
            return _FakeCurrentView._Leaf(self._select()[key])
        raise KeyError(key)

    def _as(self, which: str) -> "_FakeCurrentView":
        v = _FakeCurrentView(self._pd, self._nd)
        v._which = which
        return v

    def get_temporal_contributions(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs.get("hide_source_types", "auto") is None, (
            "adapter must pass hide_source_types=None (trap 1)"
        )
        order = sorted(self._pd, key=lambda k: -float(np.abs(self._pd[k]).max()))
        return self._as("pd"), self._as("nd"), order[:2], order


def _main() -> int:
    import time

    t0 = time.perf_counter()
    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        status = "PASS" if condition else "FAIL"
        if not condition:
            ok = False
        print(f"  [{status}] {label}" + (f"  -- {detail}" if detail else ""))

    rng = np.random.default_rng(20260915)
    angles = KNOWN_TUNING_ANGLES_DEG

    print("=" * 78)
    print("stability.py smoke test -- synthetic data, CPU")
    print("PRIMARY: M2 = current decomposition (D4).  PRIMARY TEST: PERMDISP (D3).")
    print("=" * 78)

    # ---- 1. ground-truth constants recompute from the Maisak curves ---------
    print("\n[1] Ground-truth DSI constants recomputed from KNOWN_TUNING_CURVES")
    gt = known_tuning_curves(T4T5_CELL_TYPES, angles)          # (8, 12)
    gt_peaks = gt[:, None, None, :]                            # (8, I=1, W=1, A=12)
    dsi_vs = direction_selectivity_index(gt_peaks, angles)[:, 0]
    dsi_cl = direction_selectivity_index_classic(gt, angles)
    for ct, v, c in zip(T4T5_CELL_TYPES, dsi_vs, dsi_cl):
        print(f"      {ct}: vector_sum={v:.4f} (const {KNOWN_DSI_VECTOR_SUM[ct]:.4f})"
              f"   classic={c:.4f} (const {KNOWN_DSI_CLASSIC[ct]:.4f})")
    check(
        "KNOWN_DSI_VECTOR_SUM matches recomputation",
        np.allclose(dsi_vs, [KNOWN_DSI_VECTOR_SUM[c] for c in T4T5_CELL_TYPES], atol=1e-3),
        f"max|delta|={np.max(np.abs(dsi_vs - [KNOWN_DSI_VECTOR_SUM[c] for c in T4T5_CELL_TYPES])):.2e}",
    )
    check(
        "KNOWN_DSI_CLASSIC matches recomputation",
        np.allclose(dsi_cl, [KNOWN_DSI_CLASSIC[c] for c in T4T5_CELL_TYPES], atol=1e-3),
    )
    check(
        "vector-sum DSI runs well below classic DSI on the same curves",
        bool(np.all(dsi_vs < dsi_cl)),
        f"means {dsi_vs.mean():.3f} vs {dsi_cl.mean():.3f}",
    )
    pd_gt = preferred_direction(gt, angles)
    pd_err = preferred_direction_correspondence(pd_gt, T4T5_CELL_TYPES)
    check(
        "literature curves recover canonical preferred directions (<= 30 deg)",
        pd_err["mean_abs_error_deg"] <= 30.0,
        f"mean |error| = {pd_err['mean_abs_error_deg']:.1f} deg",
    )

    # ---- 2. PRIMARY mechanism metric M2: current decomposition (D4) --------
    print("\n[2] PRIMARY mechanism metric M2 = presynaptic current decomposition")
    spec = PipelineSpec()                       # 12 presyn types x {PD, ND} = 24 dims
    cal = calibrate_sigma_for_variance_ratio(3.0, spec=spec)
    print(f"      generator calibrated: sigma_A={cal['sigma_a']:.4f} "
          f"sigma_B={cal['sigma_b']:.4f} -> realized V_B/V_A = "
          f"{cal['realized_ratio']:.4f} (target 3.0, n_ref={int(cal['n_ref'])})")
    N = 28
    mech_a = synthetic_current_ensemble(rng, N, spec=spec, sigma=cal["sigma_a"],
                                        arm="constrained")
    mech_b = synthetic_current_ensemble(rng, N, spec=spec, sigma=cal["sigma_b"],
                                        arm="baseline")
    print(f"      contributions: {mech_a.contributions.shape} per arm "
          f"(models x presynaptic types x {mech_a.conditions}), N={N} seeds/arm")
    fp_a = mechanism_fingerprint(mech_a.contributions)
    check("M2 fingerprint is jointly L1-normalized (sum|m| == 1 per model)",
          np.allclose(np.abs(fp_a).sum(axis=1), 1.0, atol=1e-12),
          f"dim {fp_a.shape[1]} = {mech_a.contributions.shape[1]} types x "
          f"{mech_a.contributions.shape[2]} conditions")
    check("M2 fingerprint keeps the SIGN of inhibitory currents (not rectified)",
          bool((fp_a < 0).any() and (fp_a > 0).any()),
          f"{int((fp_a[0] < 0).sum())} of {fp_a.shape[1]} entries negative in model 0")

    d_self = mechanism_distance(mech_a.model(0), mech_a.model(0), mech_a.source_types)
    d01 = mechanism_distance(mech_a.model(0), mech_a.model(1), mech_a.source_types)
    check("self-distance is zero", abs(d_self.value) < 1e-12)
    check("per-source-type breakdown decomposes the euclidean distance exactly",
          abs(math.sqrt(sum(v ** 2 for v in d01.per_source_type.values())) - d01.value) < 1e-12,
          f"D(A0,A1)={d01.value:.5f}")
    shuffled = list(mech_a.source_types)[::-1]
    d_shuf = mechanism_distance(
        mech_a.model(0), mech_a.model(1)[::-1], mech_a.source_types, shuffled
    )
    check("mechanism distance is matched by NAME, not by position",
          abs(d_shuf.value - d01.value) < 1e-12,
          f"{d_shuf.value:.6f} vs {d01.value:.6f}")
    D_mech = mechanism_distance_matrix(mech_a)
    check("distance matrix agrees with the pairwise loop",
          abs(D_mech[0, 1] - d01.value) < 1e-12 and abs(D_mech[3, 3]) == 0.0)
    try:
        mechanism_distance_matrix(TuningEnsemble(np.zeros((2, 8, 12)), T4T5_CELL_TYPES, angles))
        mech_guard = False
    except TypeError:
        mech_guard = True
    check("mechanism_distance_matrix REFUSES a TuningEnsemble (mechanism != response)",
          mech_guard)

    disp_mech_a = across_seed_dispersion(mech_a)
    disp_mech_b = across_seed_dispersion(mech_b)
    print("      " + disp_mech_a.summary())
    print("      " + disp_mech_b.summary())
    ratio_obs = disp_mech_b.total_variance / disp_mech_a.total_variance
    check("V(baseline)/V(constrained) recovers the calibrated 3x ratio",
          1.5 <= ratio_obs <= 6.0, f"observed {ratio_obs:.2f} at N={N}/arm")
    check("dispersion result is LABELLED as a mechanism metric",
          disp_mech_a.kind == "mechanism (currents)", disp_mech_a.kind)

    # the flyvis adapter, against a stub that reproduces the two documented traps
    dt = 1.0 / 200.0
    n_t = 40
    names = ["Mi1", "Tm3", "Mi4", "Mi9", "CT1(M10)"]
    amps_pd = {nm: float(a) for nm, a in zip(names, [1.0, 0.7, -0.4, -0.25, -0.1])}
    amps_nd = {nm: 0.5 * v for nm, v in amps_pd.items()}
    traces_pd = {nm: np.full((3, 1, n_t), v) for nm, v in amps_pd.items()}
    traces_nd = {nm: np.full((3, 1, n_t), v) for nm, v in amps_nd.items()}
    fake = _FakeCurrentView(traces_pd, traces_nd)
    ens_from_flyvis = CurrentEnsemble.from_flyvis_current_view(
        fake, names, contrast=1.0, angle=90.0, dt=dt, t_start=0.0, t_end=1.0, arm="x"
    )
    expected = (n_t - 1) * dt
    check("from_flyvis_current_view integrates each trace over time",
          np.allclose(ens_from_flyvis.contributions[:, :, 0],
                      np.array([amps_pd[nm] for nm in names]) * expected)
          and np.allclose(ens_from_flyvis.contributions[:, :, 1],
                          np.array([amps_nd[nm] for nm in names]) * expected),
          f"constant trace of {n_t} samples at dt={dt} -> integral x {expected:.3f}")
    check("adapter keeps the PRE-REGISTERED source-type set and order "
          "(ignores flyvis's contribution-sorted, truncated list)",
          ens_from_flyvis.source_types == tuple(names)
          and ens_from_flyvis.contributions.shape == (3, 5, 2),
          f"flyvis would have returned {fake.get_temporal_contributions(hide_source_types=None)[2]}")
    check("adapter calls flyvis with hide_source_types=None (trap 1)",
          fake.calls[0]["hide_source_types"] is None)

    # ---- 3. SECONDARY response metric R1: angular tuning curves ------------
    print("\n[3] SECONDARY response metric R1 (tuning curves) -- NOT a mechanism metric")
    arm_a = _synthetic_arm(rng, N, shape_noise=0.035, angle_jitter_deg=6.0, arm="constrained")
    arm_b = _synthetic_arm(rng, N, shape_noise=0.090, angle_jitter_deg=18.0, arm="baseline")
    print(f"      curves: {arm_a.curves.shape} per arm "
          f"(models x cell types x angles), N={N} seeds/arm")

    d01r = tuning_curve_response_distance(
        arm_a.model(0), arm_a.model(1), arm_a.cell_types, arm_a.cell_types
    )
    d_selfr = tuning_curve_response_distance(
        arm_a.model(0), arm_a.model(0), arm_a.cell_types, arm_a.cell_types
    )
    d_crossr = tuning_curve_response_distance(
        arm_a.model(0), arm_b.model(0), arm_a.cell_types, arm_b.cell_types
    )
    print(f"      D(A0, A1)={d01r.value:.4f}   D(A0, A0)={d_selfr.value:.2e}   "
          f"D(A0, B0)={d_crossr.value:.4f}")
    check("self-distance is zero", abs(d_selfr.value) < 1e-12)
    check("distances are non-negative and bounded by 1 (correlation metric)",
          0.0 <= d01r.value <= 1.0 and 0.0 <= d_crossr.value <= 1.0)
    check("per-cell-type breakdown is returned for all 8 matched types",
          tuple(d01r.per_cell_type) == T4T5_CELL_TYPES)

    reordered = TuningEnsemble(
        arm_a.curves[:, ::-1, :], tuple(reversed(T4T5_CELL_TYPES)), angles
    )
    d_reordered = tuning_curve_response_distance(
        arm_a.model(0), reordered.model(1), arm_a.cell_types, reordered.cell_types
    )
    check("distance is invariant to cell-type ordering (matched by name)",
          abs(d_reordered.value - d01r.value) < 1e-12,
          f"{d_reordered.value:.6f} vs {d01r.value:.6f}")

    disp_a = across_seed_dispersion(arm_a)
    disp_b = across_seed_dispersion(arm_b)
    print("      " + disp_a.summary())
    print("      " + disp_b.summary())
    check("unstable arm has larger total response variance",
          disp_b.total_variance > disp_a.total_variance,
          f"{disp_b.total_variance:.4g} > {disp_a.total_variance:.4g}")
    check("response dispersion is LABELLED as a response metric",
          disp_a.kind == "response (tuning curves)", disp_a.kind)

    fps = response_fingerprint(arm_a.curves)
    v_trace = float(np.trace(np.cov(fps.T, ddof=0)))
    iu = np.triu_indices(N, k=1)
    sq = ((fps[iu[0]] - fps[iu[1]]) ** 2).sum(axis=1)
    v_pairwise = float(sq.sum() / (N ** 2))  # (1/2N^2) * sum over ALL ordered pairs
    check("total variance identity: mean||m-mbar||^2 == tr Cov == pairwise form",
          abs(disp_a.total_variance - v_trace) < 1e-10
          and abs(disp_a.total_variance - v_pairwise) < 1e-10,
          f"{disp_a.total_variance:.10g} / {v_trace:.10g} / {v_pairwise:.10g}")

    t4c = T4T5_CELL_TYPES.index("T4c")  # the cell type both papers cluster on
    theta_a = np.array([preferred_direction(arm_a.model(s), angles)[t4c] for s in range(N)])
    theta_b = np.array([preferred_direction(arm_b.model(s), angles)[t4c] for s in range(N)])
    circ_a, circ_b = circular_dispersion(theta_a), circular_dispersion(theta_b)
    print(f"      T4c circular SD: arm A {np.rad2deg(circ_a['circular_sd_rad']):.2f} deg, "
          f"arm B {np.rad2deg(circ_b['circular_sd_rad']):.2f} deg")
    check("unstable arm has larger circular dispersion of T4c preferred direction",
          circ_b["circular_variance"] > circ_a["circular_variance"])

    # ---- 4. correspondence to biology --------------------------------------
    print("\n[4] Correspondence to biology (DSI vs Maisak 2013 targets)")
    model_dsi_a = np.array([
        direction_selectivity_index(arm_a.model(s)[:, None, None, :], angles)[:, 0]
        for s in range(N)
    ])
    corr_first = dsi_correspondence(model_dsi_a[0], T4T5_CELL_TYPES)
    print(f"      model 0: MAE vs vector-sum targets = {corr_first.mean_absolute_error:.4f}, "
          f"r = {corr_first.pearson_r:.3f}")
    check("model DSIs land in the documented vector-sum range (0.15-0.75)",
          bool(np.all(model_dsi_a > 0.15) and np.all(model_dsi_a < 0.75)),
          f"observed [{model_dsi_a.min():.3f}, {model_dsi_a.max():.3f}]")
    fake_non_ds = {ct: float(v) for ct, v in zip(NO_MOTION_TUNING_TYPES,
                                                 rng.uniform(0.02, 0.12, len(NO_MOTION_TUNING_TYPES)))}
    dsi_map = {**corr_first.as_dict(), **fake_non_ds}
    rho = dsi_correlation_to_known(dsi_map)
    print(f"      dsi_correlation_to_known over 18 labelled types: r = {rho:.3f}")
    check("DS/non-DS ordering is recovered (r > 0.8)", rho > 0.8)

    # The max-over-intensity normalization quirk, demonstrated deterministically.
    # Two intensities whose curves differ ONLY by a factor of 0.25: a
    # per-intensity normalization is scale invariant and would return the SAME
    # DSI for both; flyvis shares the larger denominator, so the weak intensity's
    # DSI is deflated by exactly that factor.
    t4c_curve = gt[t4c]
    peaks_2i = np.stack([0.25 * t4c_curve, t4c_curve])[None, :, None, :]  # (1,I=2,W=1,A)
    dsi_2i = direction_selectivity_index(peaks_2i, angles)[0]
    dsi_sep = float(np.abs((t4c_curve * np.exp(1j * np.deg2rad(angles))).sum())
                    / np.abs(t4c_curve).sum())
    print(f"      max-over-intensity quirk: DSI(weak)={dsi_2i[0]:.4f}, "
          f"DSI(strong)={dsi_2i[1]:.4f}, per-intensity norm would give "
          f"{dsi_sep:.4f} for both")
    check("flyvis max-over-intensity normalization deflates the weak contrast",
          abs(dsi_2i[0] - 0.25 * dsi_sep) < 1e-9 and abs(dsi_2i[1] - dsi_sep) < 1e-9)

    fri = flash_response_index(np.array([1.0, 0.2]), np.array([0.1, 0.9]))
    check("FRI is +1/-1 at the ON/OFF extremes",
          bool(fri[0] > 0.6 and fri[1] < -0.6), f"FRI = {np.round(fri, 3).tolist()}")

    # ---- 5. SECONDARY scalar variance tests --------------------------------
    print("\n[5] SECONDARY scalar tests on per-seed deviations (mean_pairwise centre)")
    bf = brown_forsythe_variance_test(disp_b.deviations, disp_a.deviations)
    print("      " + bf.summary())
    lev = stats.levene(disp_b.deviations, disp_a.deviations, center="median")
    check("our Brown-Forsythe == scipy.stats.levene(center='median')",
          abs(bf.statistic - float(lev.statistic)) < 1e-9
          and abs(bf.pvalue - float(lev.pvalue)) < 1e-12,
          f"scipy: stat={float(lev.statistic):.6f} p={float(lev.pvalue):.3e}")
    check("unstable arm is detected as more dispersed (p < 0.05)", bf.pvalue < 0.05)

    perm = permutation_variance_test(disp_b.deviations, disp_a.deviations,
                                     n_perm=4999, rng=np.random.default_rng(7))
    print("      " + perm.summary())
    check("permutation test agrees with Brown-Forsythe", perm.pvalue < 0.05)

    ci = log_variance_ratio_ci(disp_b.deviations, disp_a.deviations,
                               n_boot=4000, rng=np.random.default_rng(11))
    print(f"      effect size: var ratio {ci['var_ratio']:.2f} "
          f"[{ci['ratio_ci_low']:.2f}, {ci['ratio_ci_high']:.2f}] (BCa 95%)")
    check("BCa CI for the variance ratio excludes 1", ci["ratio_ci_low"] > 1.0)

    # ---- 6. false-positive rate on NULL scalar data ------------------------
    print("\n[6] Type-I error of the SCALAR tests on NULL data, alpha = 0.05")
    print("      (these hold for INDEPENDENT scalars; section [8] shows what")
    print("       happens when the same test is fed distance-to-centre values)")
    n_null, n_sims = 40, 2000
    for dist, expected_bf in (("normal", 0.05), ("t5", 0.05), ("lognormal", 0.05)):
        r = np.random.default_rng(1234)
        A = _unit_variance_variates(r, (n_sims, n_null), dist)
        B = _unit_variance_variates(r, (n_sims, n_null), dist)
        _, p_bf = _bf_pvalues(A, B)
        fpr_bf = float(np.mean(p_bf < 0.05))
        va, vb = A.var(axis=1, ddof=1), B.var(axis=1, ddof=1)
        f = va / vb
        p_f = 2.0 * np.minimum(stats.f.cdf(f, n_null - 1, n_null - 1),
                               stats.f.sf(f, n_null - 1, n_null - 1))
        fpr_f = float(np.mean(p_f < 0.05))
        print(f"      {dist:<10} Brown-Forsythe FPR = {fpr_bf:.4f}   "
              f"F-test FPR = {fpr_f:.4f}")
        check(f"Brown-Forsythe holds nominal size under {dist}",
              0.03 <= fpr_bf <= 0.075, f"{fpr_bf:.4f} (expected ~{expected_bf})")
    check("the F-test is demonstrably NOT usable (inflated under heavy tails)",
          fpr_f > 0.10, f"lognormal F-test FPR = {fpr_f:.4f}")

    # ---- 7. scalar power helper (reference only) ---------------------------
    print("\n[7] Scalar power helper -- REFERENCE ONLY, not the study's endpoint")
    pw50 = variance_test_power(50, 3.0, n_sims=3000, rng=np.random.default_rng(5))
    print(f"      power(n=50/arm, variance ratio 3.0, normal) = {pw50:.3f} "
          f"(reference ~0.93)")
    check("power at n=50 for a 3x variance ratio is in [0.85, 0.99]",
          0.85 <= pw50 <= 0.99)
    res2 = n_per_arm_for_variance_ratio(
        2.0, n_sims=1200, n_max=300, rng=np.random.default_rng(13)
    )
    print("      " + res2.summary() + f"  (reference {N_PER_ARM_REFERENCE_BF_NORMAL[2.0]})")
    check("n_per_arm for a 2x variance ratio brackets the reference value of 80",
          res2.n_per_arm is not None and 55 <= res2.n_per_arm <= 120)
    res3 = n_per_arm_for_variance_ratio(
        3.0, dist="t5", n_sims=1200, n_max=300, rng=np.random.default_rng(17)
    )
    print("      " + res3.summary() + "  (reference 48 for t_5)")
    check("heavy-tailed metric needs more seeds than the normal case",
          res3.n_per_arm is not None and res3.n_per_arm > 24)

    # ---- 8. D3 EVIDENCE: BF-on-medoid is miscalibrated for this endpoint ----
    print("\n[8] D3: WHY Brown-Forsythe on medoid deviations is NOT the primary test")
    lev_resp = medoid_leverage(fingerprint_distance_matrix(response_fingerprint(arm_b.curves)))
    spec96 = PipelineSpec(n_source_types=8, n_conditions=12, effective_rank=96.0)
    z96 = _pipeline_latents(spec96, np.random.default_rng(101), (48,))
    lev_96 = medoid_leverage(fingerprint_distance_matrix(
        _fingerprints_from_latents(spec96, z96, 0.35)))
    lev_mech = medoid_leverage(D_mech)
    for label, lv in (("tuning-curve response fp, p=96", lev_resp),
                      ("synthetic isotropic fp, p=96", lev_96),
                      ("primary current fp, p=24, p_eff=4", lev_mech)):
        print(f"      {label:<36} medoid share of Levene denominator = "
              f"{lv['share_of_within_ss']:.3f}  "
              f"(z_medoid={lv['z_medoid']:.4f} vs mean z_others="
              f"{lv['z_mean_others']:.4f}, CV of distances={lv['cv_of_distances']:.3f})")
    check("the medoid's own zero dominates Levene's denominator at p=96",
          lev_resp["share_of_within_ss"] > 0.5 and lev_96["share_of_within_ss"] > 0.5,
          "one deterministic term carries most of the within-arm SS")

    print("      size and power THROUGH THE ACTUAL PIPELINE (300 sims, n_perm=199):")
    print(f"      {'setting':<26}{'N/arm':>6}{'ratio':>7}{'PERMDISP':>10}"
          f"{'BF-medoid':>11}{'BF-meanpair':>13}")
    bf_rows = {}
    for label, sp, n_arm in (("isotropic p=96", spec96, 24),
                             ("primary p=24, p_eff=4", spec, 24)):
        for ratio in (1.0, 3.0):
            res = simulate_permdisp_power(
                n_arm, ratio, spec=sp, n_sims=300, n_perm=199,
                rng=np.random.default_rng(202 + int(ratio * 10)),
                also_brown_forsythe=True,
            )
            bf_rows[(label, ratio)] = res
            print(f"      {label:<26}{n_arm:>6}{res['var_ratio_realized']:>7.2f}"
                  f"{res['power']:>10.3f}{res['power_bf_medoid']:>11.3f}"
                  f"{res['power_bf_mean_pairwise']:>13.3f}")
    size96 = bf_rows[("isotropic p=96", 1.0)]
    pow96 = bf_rows[("isotropic p=96", 3.0)]
    check("BF-on-medoid sits FAR BELOW its nominal 0.05 size at p=96",
          size96["power_bf_medoid"] <= 0.02,
          f"{size96['power_bf_medoid']:.3f} vs nominal 0.05 "
          f"(PERMDISP {size96['power']:.3f})")
    check("BF-on-medoid has essentially NO power at a true 3x ratio where "
          "PERMDISP has full power",
          pow96["power_bf_medoid"] <= 0.25 and pow96["power"] >= 0.90,
          f"BF {pow96['power_bf_medoid']:.3f} vs PERMDISP {pow96['power']:.3f} "
          f"at N=24/arm, p=96")

    # ---- 9. D3: PERMDISP as the primary test -------------------------------
    print("\n[9] D3: PERMDISP is the PRIMARY test -- validation, calibration, N")
    fps_pool = np.concatenate([
        mechanism_fingerprint(mech_a.contributions),
        mechanism_fingerprint(mech_b.contributions),
    ])
    groups = np.array(["A"] * N + ["B"] * N)
    D_eu = fingerprint_distance_matrix(fps_pool, metric="euclidean")
    D_co = fingerprint_distance_matrix(fps_pool, metric="correlation")
    for label, Dm in (("euclidean (a true metric)", D_eu),
                      ("correlation (NOT a metric: negative eigenvalues)", D_co)):
        G = _gower(Dm)
        member = np.stack([(groups == lab).astype(float) for lab in ("A", "B")])
        z_gram = _dispersions_centroid_from_gram(G, member)
        real, imag = _pcoa(Dm)
        z_pcoa = _dispersions(real, imag, groups, "centroid")
        print(f"      {label:<48} max|z_gram - z_pcoa| = "
              f"{np.max(np.abs(z_gram - z_pcoa)):.2e}  "
              f"(imag axes: {imag.shape[1]})")
        check(f"Gower quadratic form == PCoA transcription [{label.split(' ')[0]}]",
              np.max(np.abs(z_gram - z_pcoa)) < 1e-10)
    z_direct = np.concatenate([
        np.linalg.norm(fps_pool[:N] - fps_pool[:N].mean(axis=0), axis=1),
        np.linalg.norm(fps_pool[N:] - fps_pool[N:].mean(axis=0), axis=1),
    ])
    G_eu = _gower(D_eu)
    member = np.stack([(groups == lab).astype(float) for lab in ("A", "B")])
    z_eu = _dispersions_centroid_from_gram(G_eu, member)
    check("for euclidean D the PERMDISP z equals ||x - group centroid||",
          np.max(np.abs(z_eu - z_direct)) < 1e-10,
          f"max|delta| = {np.max(np.abs(z_eu - z_direct)):.2e}")
    check("mean z^2 equals ensemble_total_variance exactly (centroid choice)",
          abs(float((z_eu[:N] ** 2).mean()) - disp_mech_a.total_variance) < 1e-12,
          f"{float((z_eu[:N] ** 2).mean()):.8g} vs {disp_mech_a.total_variance:.8g}")
    f_scipy = float(stats.f_oneway(z_eu[:N], z_eu[N:]).statistic)
    pd_obs = permdisp(D_eu, groups, center="centroid", n_perm=999,
                      rng=np.random.default_rng(23))
    print("      " + pd_obs.summary())
    check("PERMDISP statistic == scipy.stats.f_oneway on the same z",
          abs(pd_obs.statistic - f_scipy) < 1e-9, f"scipy F = {f_scipy:.6f}")
    check("PERMDISP detects the 3x mechanism-dispersion difference (p < 0.05)",
          pd_obs.pvalue < 0.05)
    pd_med = permdisp(D_eu, groups, center="median", n_perm=299,
                      rng=np.random.default_rng(29))
    print("      robustness: " + pd_med.summary())
    check("PERMDISP with the spatial median agrees with the centroid version",
          pd_med.pvalue < 0.05)
    pdisp_resp = permdisp(_pooled_distance_matrix(arm_a, arm_b), groups,
                          center="centroid", n_perm=999, rng=np.random.default_rng(3))
    print("      secondary response metric: " + pdisp_resp.summary())
    check("PERMDISP also detects heterogeneity in the SECONDARY response metric",
          pdisp_resp.pvalue < 0.05)

    print("      false-positive rate on NULL data, through the ACTUAL pipeline")
    print("      (identical arms; 400 sims, N=24/arm, n_perm=199, alpha=0.05):")
    for dist in ("normal", "t5", "lognormal"):
        sp = PipelineSpec(dist=dist)
        res = simulate_permdisp_power(24, 1.0, spec=sp, n_sims=400, n_perm=199,
                                      rng=np.random.default_rng(31))
        print(f"      {dist:<10} PERMDISP FPR = {res['power']:.4f}")
        check(f"PERMDISP holds its nominal size under {dist}",
              0.025 <= res["power"] <= 0.08,
              f"{res['power']:.4f} (+-{1.96 * math.sqrt(0.05 * 0.95 / 400):.4f} MC 95%)")

    print("      re-derived N per arm (end-to-end, not the scalar table):")
    for key in sorted(PIPELINE_POWER_TABLE_PERMDISP):
        row = PIPELINE_POWER_TABLE_PERMDISP[key]
        print(f"      {key:<20} " + "  ".join(
            f"{r:g}x:{('>300' if row[r] is None else row[r])}" for r in sorted(row)))
    check_spec = PipelineSpec(effective_rank=1.0)
    recomputed = n_per_arm_for_permdisp(
        3.0, spec=check_spec, n_sims=200, n_perm=199,
        n_grid=(12, 16, 20, 24, 30, 36, 48, 64), rng=np.random.default_rng(37),
    )
    ref = PIPELINE_POWER_TABLE_PERMDISP["normal/p_eff=1"][3.0]
    print("      recomputed " + recomputed.summary())
    check("the shipped N table is reproduced (worst-case p_eff=1, 3x ratio)",
          recomputed.n_per_arm is not None and abs(recomputed.n_per_arm - ref) <= max(8, 0.4 * ref),
          f"recomputed {recomputed.n_per_arm} vs table {ref} "
          f"(MC wobble; n_sims={recomputed.n_sims} here vs 400 for the table)")

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 78)
    print(f"{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'} in {elapsed:.2f} s")
    print("=" * 78)
    return 0 if ok else 1


def _pooled_distance_matrix(arm_a: TuningEnsemble, arm_b: TuningEnsemble) -> np.ndarray:
    """Distance matrix over the concatenation of two RESPONSE arms (for PERMDISP)."""
    pooled = TuningEnsemble(
        np.concatenate([arm_a.curves, arm_b.curves], axis=0),
        arm_a.cell_types,
        arm_a.angles_deg,
    )
    # Euclidean on L1-normalized fingerprints is a true metric, which is what
    # PERMDISP's PCoA expects; correlation distance would introduce negative
    # eigenvalues that Anderson's correction then has to absorb.
    return response_distance_matrix(pooled, metric="euclidean", normalize="l1")


if __name__ == "__main__":
    raise SystemExit(_main())
