"""Dale-constrained, type-tied, liquid-time-constant parameterization for
connectome-constrained recurrent networks (flyvis-compatible).

This module is the scientific core of the project. It is a *drop-in replacement
for the parameterization only*: same connectome, same task, same training loop,
same decoder. Nothing here trains, loads, saves, or plots anything.

===============================================================================
D1 -- THE SHARED SUBSTRATE (read this before anything else)
===============================================================================

**Both experimental arms are built by the same constructor, from the same code
path, and differ in exactly three declared axes.** Use :func:`build_arm`::

    baseline    = build_arm(con, dale=False, tying="per_type_pair", liquid_tau=False)
    constrained = build_arm(con, dale=True,  tying="per_type_pair", liquid_tau=True)

Everything else is SHARED SUBSTRATE and is not reachable through any switch:

  * **Softplus magnitude parameterization.** ``magnitude = softplus(raw)`` in
    BOTH arms. There is no clamp, no projection, no alternative
    parameterization, and no keyword that could turn it off -- see
    :attr:`TypedWeights.MAGNITUDE_PARAMETERIZATION`.
  * **The exact-exponential (zero-order-hold) integrator** in BOTH arms
    (D5) -- see :attr:`DaleTypedCfCCell.INTEGRATOR`.
  * Initialization (magnitude ``rho/<N>``, ``tau_init``, resting-potential draw,
    seed handling), activation, synapse-count factor, and the fixed/learnable
    split.

Why this matters -- the killer objection (science-1). In the 50 released flyvis
models, 418 of 604 ``edges_syn_strength`` entries are exactly 0.0 in some seeds
and non-zero in others, and those entries carry 69.2% of the across-seed
variance of ``edges_syn_strength`` (measured by the round-1 reviewer on the
released checkpoints; NOT re-measured here -- see VERIFICATION LEDGER below).
That point mass at zero is manufactured by flyvis's projected-gradient clamp
(``param.data.clamp_(0)`` at ``Network.forward``, verified this session at
network.py:496/527), not by biology. Softplus removes the point mass *by
construction*. If softplus sat only in the treatment arm, the headline
"constrained models are less variable" result would be pre-explained by the
projection operator before Dale's law did any work. So softplus is substrate,
not treatment.

Likewise the optimizer geometry (science-2). flyvis trains with Adam at
lr 5e-5 (verified this session in ``flyvis/config/scheduler/scheduler.yaml``),
and Adam's raw step is gradient-scale-invariant, so the SEMANTIC step through
softplus is ``lr * sigmoid(raw) = lr * (1 - exp(-magnitude))``. A softplus-only
treatment arm would move its magnitudes ~200x more slowly than a clamp baseline
at flyvis's own init. With softplus in both arms that Jacobian is identical by
construction, and :func:`assert_shared_substrate` checks it element-wise.

THE THREE TREATMENT AXES (and nothing else):

  (a) ``dale``      -- sign rule. ``False`` = flyvis's rule, one fixed sign per
                       ``(pre_type, post_type)`` pair, taken from the
                       connectome. ``True`` = Dale's law, one fixed sign per
                       PRESYNAPTIC cell type, applied to every edge that type
                       emits.
  (b) ``tying``     -- magnitude tying granularity: ``per_edge`` /
                       ``per_type_pair`` (flyvis default) / ``per_post_type``.
  (c) ``liquid_tau``-- whether the per-cell-type time constant is
                       input-dependent (CfC gate) or static. The integrator is
                       the same exact exponential either way.

:func:`assert_one_variable_apart` refuses any pair of arms that differs in more
than one axis, and :func:`assert_shared_substrate` proves, tensor by tensor,
that two constructed arms share everything else.

===============================================================================
NOVELTY LEDGER -- what is inherited from flyvis vs. what is new here
===============================================================================

flyvis (Lappalainen et al., *Nature* 634:1132-1140, 2024; code at
TuragaLab/flyvis @ v1.2.0, commit 92b3845) ALREADY does the following. None of
it is claimed as novel and none of it is reimplemented here:

  * Connectome-constrained sparse recurrent dynamics on a FIXED topology
    (45,669 neurons / 1,513,231 edges from FIB-25/FIB-19). INHERITED.
  * **Parameter tying per cell type.** This is flyvis's central contribution
    and its headline "734 free parameters" claim. Specifically:
      - `nodes_bias`       -> 65 values, one per cell type          (learnable)
      - `nodes_time_const` -> 65 values, one per cell type          (learnable)
      - `edges_sign`       -> 604 values, per (source_type, target_type)  (FIXED)
      - `edges_syn_count`  -> 2355 values, per (pair, du, dv)            (FIXED)
      - `edges_syn_strength`-> 604 values, per (source_type, target_type)(learnable)
    In particular **tying one weight per (pre_type, post_type) pair is ALREADY
    THE FLYVIS DEFAULT** for `sign` and `syn_strength`. Do not claim it.
  * Weight factorization `w = sign * syn_count * syn_strength`. INHERITED.
  * Non-negative synaptic magnitude. INHERITED -- but flyvis enforces it by
    *projected gradient descent* (`param.data.clamp_(0)` at the start of every
    forward pass), not by reparametrization.
  * Learnable per-cell-type time constants, initialized to 50 ms. INHERITED.

What is DIFFERENT here, in two categories:

  SUBSTRATE CHANGES -- applied to BOTH arms, therefore never a claim:
    (S1) Softplus magnitudes instead of clamp-based projected gradient descent.
    (S2) Exact-exponential (zero-order-hold) integration instead of explicit
         forward Euler with the ``1/max(tau, dt)`` guard.
    Both are methodological repairs of the baseline, not results. Any write-up
    must present them as such and must report the baseline arm trained WITH
    them, never flyvis's released models, as the control.

  TREATMENT AXES -- the only things the experiment varies:
    (N1) DALE'S LAW PROPER: one sign per **presynaptic cell type**, applied to
         every edge that neuron emits. flyvis is *not* Dale-constrained: its
         signs are per (pre, post) *pair*, and in the shipped connectome four
         source types emit both signs -- exactly {Am, C2, C3, R8}, verified this
         session by parsing ``flyvis/connectome/fib25-fib19_v2.2.json`` at
         commit 92b3845. Their signs come from published neurotransmitter
         identity (see :data:`PUBLISHED_NT_SIGNS`), not from a vote (D6).
    (N2) TYPED TYING AS A CONTROLLED SWITCH: per-edge / per-type-pair /
         per-post-type, so the experiment can measure what tying costs in task
         error. The pair granularity itself is flyvis's, not ours.
    (N3) PER-TYPE LIQUID TIME CONSTANTS via the closed-form continuous-time
         (CfC) formulation: tau becomes input-dependent, with one gate per
         **cell type**. ``grep -rni "liquid|cfc|ltc|closed.form"`` over the
         flyvis source returns ZERO hits.

===============================================================================
D5 -- THE INTEGRATOR IS SHARED, AND THE OLD "O(dt^2)" CLAIM WAS FALSE
===============================================================================

An earlier version of this docstring said the two models "differ only in the
integrator, which agrees to O(dt^2)". That is false at the operating point.
flyvis integrates with explicit forward Euler and a ``1/max(tau, dt)`` guard
(verified this session: ``flyvis/network/dynamics.py`` line 209 writes
``1 / torch.max(params.nodes.time_const, tensor(dt)) * (-v + bias + ... + x_t)``
and ``flyvis/network/network.py`` line 406 applies ``state + vel * dt``). At
flyvis's own operating point ``dt = 0.02 s`` and ``tau_init = 0.05 s``, so
``dt/tau = 0.4`` and the per-step retention factors are

    Euler:  1 - dt/tau      = 0.600000
    exact:  exp(-dt/tau)    = 0.670320     ->  +11.72% relative, per step

compounding over the 19-frame rollout. That is not a negligible second-order
term. Therefore **both arms use the exact exponential update**, and the only
integrator-related difference between arms is whether ``tau`` is
input-dependent. :func:`integrator_comparison` recomputes the table above and
the smoke test asserts the two numbers.

Related: in the released ensemble 73.7% of (model, cell-type) time constants sit
BELOW ``dt`` (round-1 reviewer's measurement, not re-measured here), where
flyvis's guard pins the effective tau to ``dt`` and the gradient w.r.t. tau is
identically zero. The exact update has no such regime. :func:`tau_dt_report`
exposes the learned-tau-vs-dt distribution per arm so this can be reported
rather than assumed.

===============================================================================
D7 -- PARAMETER ACCOUNTING (929, +26.6%)
===============================================================================

For the real connectome with ``tying="per_type_pair"`` and ``liquid_tau=True``:

    magnitudes                 604
    tau (raw_tau 65 + gate_v/gate_u/gate_b 3x65)   260
    bias                        65
    -------------------------------
    total learnable            929      = flyvis's 734 + 195  (+26.6%)

With ``liquid_tau=False`` the count is 604 + 65 + 65 = 734, i.e. EXACTLY
flyvis's baseline count -- which is what makes the CfC axis the only
capacity-changing axis. The parameter-matched control filler is therefore
**195** parameters, not 33. :class:`ParameterReport` is the single source of
this arithmetic (:meth:`ParameterReport.for_fly_connectome`), and
:meth:`ParameterReport.check_against_module` asserts that the report equals
``sum(p.numel() for p in cell.parameters())`` so the two can never drift apart
again.

===============================================================================
DEPENDENCIES AND THE FLYVIS SEAM
===============================================================================

The core of this module depends only on torch. flyvis is NOT imported at module
import time -- this file is importable and testable with torch alone.

Training inside flyvis goes through ``src/flyvis_seam.py``, not this module:
``flyvis_arm_config`` / ``apply_arm_to_solver_config`` build each arm's flyvis
config (SoftplusValue magnitudes with ``clamp: None``, exact integrator, tau
clamp, DaleSynapseSign, per-type CfC gates or inert filler), and
``flyvis_seam.verify_seam()`` checks the built networks on tensors. This module
remains the standalone reference implementation used by Stage 0.

VERIFICATION LEDGER
-------------------
VERIFIED THIS SESSION (by reading the pinned source / parsing the shipped data /
querying Crossref):
  * ``flyvis/network/dynamics.py``: Euler velocity with ``1/max(tau, dt)``.
  * ``flyvis/network/network.py``: ``state + vel*dt``; ``param.data.clamp_(0)``
    under ``clamp="non_negative"``, called from ``forward``; the default
    ``syn_strength`` config carries ``clamp="non_negative"``.
  * ``flyvis/config/scheduler/scheduler.yaml``: lr 5e-5 -> 5e-6, dt 0.02.
  * ``flyvis/network/initialization.py``: ``InitialDistribution.raw_values`` /
    ``semantic_values``; the ``clamp`` docstring's explicit warning that
    parameters using ``semantic_values`` "should not use clamp";
    ``SynapseCountScaling``'s ``alpha = rho/<N>``; ``SynapseSign`` grouping by
    ``["source_type", "target_type"]``; ``TimeConstant`` grouping by
    ``["type"]``.
  * ``flyvis/utils/class_utils.py``: ``forward_subclass`` finds subclasses by
    ``__qualname__`` and unpacks the remaining config keys as ``__init__``
    kwargs -- which is why the adapters below take their options from the
    config rather than from a closure.
  * ``flyvis/connectome/fib25-fib19_v2.2.json``: 605 ``(src, tar)`` rows, 65
    node types, 63 presynaptic types, mixed-sign source types exactly
    {Am, C2, C3, R8}; per-row ``alpha_references`` provenance; the
    synapse-count-weighted votes reported in :class:`DaleSignResolver`.
  * Every reference in :data:`PUBLISHED_NT_SIGNS` (title, journal, volume,
    pages, DOI via Crossref; the Kolodziejczyk and Xiao abstract sentences
    quoted verbatim from the publisher/PMC pages).

NOT verified here, carried from the round-1 review (labelled, not promoted):
  * 418/604 mixed-zero magnitudes and their 69.2% share of across-seed variance.
  * 73.7% of (model, cell-type) time constants below dt in the released
    ensemble; 57 of 65 types straddling that boundary.
  * The released checkpoints' 604 ``edges_syn_strength`` entries. NOTE the
    shipped JSON has 605 distinct ``(src, tar)`` pairs (verified this session):
    one pair is evidently not realized on the rendered hex lattice. This module
    uses 604 because that is what the trained checkpoints contain, and says so.

STILL UNKNOWN -- documented, not guessed:
  * Whether flyvis *intends* ``NetworkDynamics`` to be ``nn.Module``-
    subclassable. Undocumented and untested upstream.
  * How to make user-defined subclasses visible to the Hydra CLI
    (``flyvis train-single``). Only the Python API path is verified.
  * Checkpoint compatibility: ``recover_network`` calls ``load_state_dict`` with
    ``strict=True``, so pretrained flyvis checkpoints will NOT load into a
    re-parameterized network. Train both arms from scratch.

References
----------
CfC closed form:
  Hasani, Lechner, Amini, Liebenwein, Ray, Tschaikowski, Teschl & Rus,
  "Closed-form continuous-time neural networks", Nature Machine Intelligence
  4:992-1003 (2022); arXiv:2106.13898.
Liquid time-constant networks:
  Hasani, Lechner, Amini, Rus & Grosu, "Liquid Time-constant Networks", AAAI 2021;
  arXiv:2006.04439.
Exactness of the exponential update for linear leaky integrators:
  Rotter & Diesmann, "Exact digital simulation of time-invariant linear systems
  with applications to neuronal modeling", Biol. Cybern. 81:381-402 (1999).
Baseline model:
  Lappalainen et al., "Connectome-constrained networks predict neural activity
  across the fly visual system", Nature 634:1132-1140 (2024).
Neurotransmitter identities: see :data:`PUBLISHED_NT_SIGNS`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    # connectome
    "TypedConnectome",
    # Dale / neurotransmitters (D6)
    "NeurotransmitterAssignment",
    "PUBLISHED_NT_SIGNS",
    "AMBIGUOUS_SOURCE_TYPES",
    "SIGN_CONVENTION_REFERENCES",
    "CONNECTOME_VOTES",
    "CONNECTOME_ALPHA_REFERENCES",
    "published_explicit_signs",
    "nt_citation_table",
    "resolve_edge_signs",
    "mixed_sign_types",
    # parameterization
    "TypedWeights",
    "PerTypeLiquidTau",
    "DaleTypedCfCCell",
    "ParameterReport",
    "TyingMode",
    "SignPolicy",
    "SignRule",
    "DaleScope",
    "inverse_softplus",
    # arms (D1)
    "ArmSpec",
    "ARM_AXES",
    "BASELINE_ARM",
    "CONSTRAINED_ARM",
    "PREREGISTERED_ARMS",
    "build_arm",
    "factorial_ladder",
    "differing_axes",
    "assert_one_variable_apart",
    "assert_shared_substrate",
    # integrator / tau (D5)
    "exact_retention",
    "flyvis_euler_retention",
    "IntegratorRow",
    "integrator_comparison",
    "TauDtReport",
    "tau_dt_report",
    # honesty about what is absent
    "NOT_IMPLEMENTED",
    "missing_api",
    # constants
    "FLYVIS",
]


# =============================================================================
# verified constants
# =============================================================================


@dataclass(frozen=True)
class _FlyvisConstants:
    """Constants of the flyvis v1.2.0 baseline, each with its provenance.

    Everything here was read out of the pinned source tree or the shipped
    connectome JSON during this session (see the module docstring's
    VERIFICATION LEDGER), except where the field docstring says otherwise.
    """

    commit: str = "92b3845cc426dd309a1a0e1b3890156c42e14021"
    version: str = "v1.2.0"
    # connectome shape
    n_edges: int = 1_513_231
    n_neurons: int = 45_669
    n_types: int = 65
    n_source_types: int = 63
    # (src, tar) pairs: 605 rows in the shipped JSON, 604 realized in the
    # released checkpoints' edges_syn_strength. Both numbers are real.
    n_type_pairs_json: int = 605
    n_type_pairs_checkpoint: int = 604
    n_syn_count_groups: int = 2355
    # learnable split of the baseline: 65 bias + 65 tau + 604 syn_strength
    n_learnable: int = 734
    # training operating point
    dt: float = 0.02
    tau_init: float = 0.05
    lr_start: float = 5e-5
    lr_stop: float = 5e-6
    magnitude_scale: float = 0.01
    bias_mean: float = 0.5
    bias_std: float = 0.05


FLYVIS = _FlyvisConstants()

# Tying granularity. The experiment switches over these to measure what tying
# costs in task error.
#   "per_edge"      -- one magnitude per edge (no tying; the loose control arm)
#   "per_type_pair" -- one magnitude per (pre_type, post_type)  [flyvis default]
#   "per_post_type" -- one magnitude per post_type (tightest; strong ablation)
TyingMode = Literal["per_edge", "per_type_pair", "per_post_type"]

# How a single sign is chosen for each presynaptic cell type.
SignPolicy = Literal["weighted_majority", "majority", "explicit"]

# Which sign rule an arm uses. This is treatment axis (a).
#   "connectome_pair" -- flyvis's rule: one fixed sign per (pre, post) pair.
#   "dale"            -- Dale's law: one fixed sign per presynaptic type.
SignRule = Literal["connectome_pair", "dale"]

# Which presynaptic types Dale's law is enforced on.
#   "all"               -- every presynaptic type (the pre-registered treatment).
#   "unambiguous_only"  -- only types that already emit a single sign in the
#                          connectome; the four mixed types keep their per-pair
#                          signs. This is the PRE-REGISTERED ABLATION ARM of D6,
#                          not a fallback.
DaleScope = Literal["all", "unambiguous_only"]

_EPS = 1e-12


# =============================================================================
# numerics
# =============================================================================


def inverse_softplus(y: Tensor, eps: float = 1e-6) -> Tensor:
    """Numerically stable inverse of ``softplus``.

    ``softplus(inverse_softplus(y)) == y`` for ``y > 0``.

    The naive form ``log(expm1(y))`` overflows ``expm1`` for ``y`` above roughly
    88 in float32. For large ``y``, ``softplus(x) ~= x``, so the identity is used
    instead. Values are clamped to ``eps`` from below because ``inverse_softplus``
    diverges at 0.

    Args:
        y: Target (strictly positive) values.
        eps: Lower clamp applied to ``y`` before inversion.

    Returns:
        Raw pre-softplus values of the same shape and dtype as ``y``.
    """
    y = y.clamp(min=eps)
    return torch.where(y > 20.0, y, torch.log(torch.expm1(y)))


# =============================================================================
# connectome interface
# =============================================================================


@dataclass
class TypedConnectome:
    """Minimal sparse connectome description consumed by this module.

    This is the clean abstract interface referred to in the module docstring. It
    is deliberately smaller than flyvis's connectome: it is exactly the fields
    the parameterization needs, nothing else. Any connectome (flyvis's, a
    degree-preserving rewired null, a synthetic test graph) can be expressed as
    one of these.

    All tensors are 1-D and live on the same device.

    Attributes:
        source_index: ``(n_edges,)`` int64. Presynaptic neuron index per edge.
        target_index: ``(n_edges,)`` int64. Postsynaptic neuron index per edge.
        neuron_type: ``(n_neurons,)`` int64. Cell-type id per neuron, in
            ``[0, n_types)``.
        type_names: Human-readable name per type id, length ``n_types``. Used for
            interpretable reporting and for downstream mechanism analysis.
        syn_count: ``(n_edges,)`` float. Synapse count per edge. Held FIXED
            (non-learnable), exactly as in flyvis. Defaults to all ones.
        edge_sign_hint: Optional ``(n_edges,)`` float in ``{-1, +1}``. The
            connectome's own per-edge sign, which in flyvis is constant within a
            ``(pre_type, post_type)`` pair. This IS the baseline arm's sign rule
            (``sign_rule="connectome_pair"``). It is also the input the Dale
            vote consumes when no published neurotransmitter sign is available.
    """

    source_index: Tensor
    target_index: Tensor
    neuron_type: Tensor
    type_names: List[str]
    syn_count: Optional[Tensor] = None
    edge_sign_hint: Optional[Tensor] = None

    def __post_init__(self) -> None:
        self.source_index = self.source_index.long()
        self.target_index = self.target_index.long()
        self.neuron_type = self.neuron_type.long()
        if self.source_index.shape != self.target_index.shape:
            raise ValueError("source_index and target_index must have equal shape")
        if self.source_index.dim() != 1:
            raise ValueError("edge index tensors must be 1-D")
        if self.neuron_type.dim() != 1:
            raise ValueError("neuron_type must be 1-D")
        if int(self.neuron_type.max()) >= len(self.type_names):
            raise ValueError("neuron_type contains an id with no entry in type_names")
        if self.syn_count is None:
            self.syn_count = torch.ones_like(self.source_index, dtype=torch.float32)
        if self.syn_count.shape != self.source_index.shape:
            raise ValueError("syn_count must have one entry per edge")
        if self.edge_sign_hint is not None:
            if self.edge_sign_hint.shape != self.source_index.shape:
                raise ValueError("edge_sign_hint must have one entry per edge")

    @property
    def n_edges(self) -> int:
        """Number of directed edges (synaptic connections) in the graph."""
        return int(self.source_index.numel())

    @property
    def n_neurons(self) -> int:
        """Number of neurons (nodes) in the graph."""
        return int(self.neuron_type.numel())

    @property
    def n_types(self) -> int:
        """Number of distinct cell types."""
        return len(self.type_names)

    @property
    def edge_source_type(self) -> Tensor:
        """``(n_edges,)`` cell-type id of the presynaptic neuron of each edge."""
        return self.neuron_type[self.source_index]

    @property
    def edge_target_type(self) -> Tensor:
        """``(n_edges,)`` cell-type id of the postsynaptic neuron of each edge."""
        return self.neuron_type[self.target_index]

    def type_id(self, name: str) -> int:
        """Return the type id of ``name``; raises ``KeyError`` if absent."""
        try:
            return self.type_names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown cell type {name!r}") from exc

    # -- adapters -------------------------------------------------------------

    @classmethod
    def from_flyvis(cls, connectome: Any) -> "TypedConnectome":
        """Build a :class:`TypedConnectome` from a flyvis connectome directory.

        Column names below were VERIFIED by reading
        ``flyvis/connectome/connectome.py`` at commit 92b3845: the node table has
        ``index, type, u, v, role`` and the edge table has ``source_index,
        target_index, sign, n_syn, source_type, target_type, ...``. Cell types are
        stored as bytes and decoded here.

        Note that the per-neuron type is derived from ``nodes.type``, not from
        ``edges.source_type``, so that types are consistent by construction.

        Args:
            connectome: A flyvis connectome object, e.g.
                ``flyvis.Network().connectome``.

        Returns:
            The equivalent :class:`TypedConnectome`.
        """
        import numpy as np

        def _decode(arr: Any) -> List[str]:
            out: List[str] = []
            for v in np.asarray(arr).reshape(-1):
                out.append(v.decode() if isinstance(v, (bytes, bytearray)) else str(v))
            return out

        node_type_names = _decode(connectome.nodes.type[:])
        # Stable, sorted type ids so runs are reproducible across machines.
        uniq = sorted(set(node_type_names))
        lookup = {name: i for i, name in enumerate(uniq)}
        neuron_type = torch.tensor([lookup[n] for n in node_type_names], dtype=torch.long)

        src = torch.as_tensor(np.asarray(connectome.edges.source_index[:]), dtype=torch.long)
        tgt = torch.as_tensor(np.asarray(connectome.edges.target_index[:]), dtype=torch.long)
        n_syn = torch.as_tensor(
            np.asarray(connectome.edges.n_syn[:]), dtype=torch.float32
        )
        sign = torch.as_tensor(np.asarray(connectome.edges.sign[:]), dtype=torch.float32)

        return cls(
            source_index=src,
            target_index=tgt,
            neuron_type=neuron_type,
            type_names=uniq,
            syn_count=n_syn,
            edge_sign_hint=sign,
        )


# =============================================================================
# D6 -- published neurotransmitter identities for the four ambiguous types
# =============================================================================

# The mixed-sign presynaptic types of the shipped flyvis connectome. VERIFIED
# this session by parsing flyvis/connectome/fib25-fib19_v2.2.json at commit
# 92b3845: of the 63 presynaptic types in the 605 (src, tar) rows, exactly these
# four emit both +1 and -1 alphas.
AMBIGUOUS_SOURCE_TYPES: Tuple[str, ...] = ("Am", "C2", "C3", "R8")

# Synapse-count-weighted vote of the shipped connectome's own per-pair alphas,
# and the unweighted (per-pair) vote, for those four types. VERIFIED this
# session from the same JSON. These are reported so that the cost of REPLACING
# the vote with published neurotransmitter identity is visible.
CONNECTOME_VOTES: Dict[str, Dict[str, float]] = {
    "Am": {"weighted_pos": 64.00, "weighted_neg": 31.00, "pairs_pos": 2, "pairs_neg": 9},
    "C2": {"weighted_pos": 29.95, "weighted_neg": 165.05, "pairs_pos": 2, "pairs_neg": 12},
    "C3": {"weighted_pos": 62.12, "weighted_neg": 365.58, "pairs_pos": 2, "pairs_neg": 20},
    "R8": {"weighted_pos": 42.80, "weighted_neg": 45.81, "pairs_pos": 4, "pairs_neg": 5},
}

# flyvis's own provenance strings for those alphas, read from the JSON's
# per-row ``alpha_references`` field (VERIFIED this session). Two of the three
# sources are personal communications, which is precisely why the vote is not an
# acceptable substitute for published neurotransmitter identity.
CONNECTOME_ALPHA_REFERENCES: Dict[str, Tuple[str, ...]] = {
    "Am": ("NernPC2018",),
    "C2": ("NernPC2018", "ReiserPC2017"),
    "C3": ("NernPC2018", "ReiserPC2017"),
    "R8": ("Hardie1989", "NernPC2018"),
}


@dataclass(frozen=True)
class NeurotransmitterAssignment:
    """A published neurotransmitter identity and the sign it implies.

    Attributes:
        cell_type: Cell-type name as it appears in the flyvis connectome.
        aliases: Other names the same cell carries in the literature.
        transmitter: The fast neurotransmitter this type releases.
        receptor: The postsynaptic receptor that determines the SIGN.
        sign: ``+1`` (depolarizing) or ``-1`` (hyperpolarizing).
        evidence: One-line statement of the evidence, quoting the source where
            the exact wording matters.
        references: Full citations, each verified against Crossref this session.
        caveat: Anything that makes the assignment less than airtight. Never
            empty for a type where it should not be.
        connectome_vote_sign: What the synapse-count-weighted vote over the
            shipped connectome would have chosen instead, for comparison.
    """

    cell_type: str
    transmitter: str
    receptor: str
    sign: float
    evidence: str
    references: Tuple[str, ...]
    caveat: str = ""
    aliases: Tuple[str, ...] = ()
    connectome_vote_sign: Optional[float] = None

    def summary(self) -> str:
        """One-paragraph human-readable justification of this sign."""
        alias = f" (aka {', '.join(self.aliases)})" if self.aliases else ""
        vote = (
            f"  connectome vote would give : {self.connectome_vote_sign:+.0f}"
            if self.connectome_vote_sign is not None
            else ""
        )
        lines = [
            f"{self.cell_type}{alias}: {self.transmitter} -> {self.receptor} "
            f"=> sign {self.sign:+.0f}",
            f"  evidence : {self.evidence}",
        ]
        if vote:
            lines.append(vote)
        if self.caveat:
            lines.append(f"  caveat   : {self.caveat}")
        for r in self.references:
            lines.append(f"  ref      : {r}")
        return "\n".join(lines)


# The sign convention used throughout flyvis and here: cholinergic types
# depolarize their targets (+1); histaminergic, GABAergic and glutamatergic
# types in the fly optic lobe hyperpolarize theirs (-1), because their canonical
# fast receptors are ligand-gated chloride channels (ort/HisCl1, Rdl, GluCl-alpha).
SIGN_CONVENTION_REFERENCES: Tuple[str, ...] = (
    "Matsliah A. et al. (2024) 'Predicting visual function by interpreting a "
    "neuronal wiring diagram.' Nature 634:113-123. doi:10.1038/s41586-024-07953-5 "
    "-- states the convention that histaminergic, GABAergic and glutamatergic "
    "neurons are assigned hyperpolarizing output while cholinergic neurons "
    "produce depolarizing output.",
    "Liu W.W. & Wilson R.I. (2013) 'Glutamate is an inhibitory neurotransmitter "
    "in the Drosophila olfactory system.' PNAS 110:10294-10299. "
    "doi:10.1073/pnas.1220560110 -- GluCl-alpha mediates fast inhibition.",
    "Eckstein N. et al. (2024) 'Neurotransmitter classification from electron "
    "microscopy images at synaptic sites in Drosophila melanogaster.' Cell "
    "187:2574-2594.e23. doi:10.1016/j.cell.2024.03.016 -- whole-brain "
    "EM-based transmitter predictions, 94% per-neuron accuracy.",
)

# D6: signs for the four mixed-sign presynaptic types, from published
# neurotransmitter assignments rather than from a vote. Every reference below
# was checked against Crossref during this session (title, journal, volume,
# pages, DOI); the two quoted sentences were read from the publisher's own
# abstract page.
PUBLISHED_NT_SIGNS: Dict[str, NeurotransmitterAssignment] = {
    "Am": NeurotransmitterAssignment(
        cell_type="Am",
        aliases=("Lai", "lamina amacrine"),
        transmitter="glutamate",
        receptor="GluCl-alpha (glutamate-gated chloride channel)",
        sign=-1.0,
        evidence=(
            "Kolodziejczyk et al. 2008 report, verbatim in the abstract, "
            "'Strong vGluT immunoreactivity was detected in alpha-processes of "
            "amacrine neurons' -- the alpha-processes are the amacrine cell's "
            "defining lamina processes. Glutamate acts through GluCl-alpha in "
            "the fly optic lobe, where it is inhibitory."
        ),
        references=(
            "Kolodziejczyk A., Sun X., Meinertzhagen I.A., Nassel D.R. (2008) "
            "'Glutamate, GABA and acetylcholine signaling components in the "
            "lamina of the Drosophila visual system.' PLoS ONE 3(5):e2110. "
            "doi:10.1371/journal.pone.0002110",
            "Molina-Obando S. et al. (2019) 'ON selectivity in the Drosophila "
            "visual system is a multisynaptic process involving both "
            "glutamatergic and GABAergic inhibition.' eLife 8:e49373. "
            "doi:10.7554/eLife.49373",
            "Fischbach K.F. & Dittrich A.P.M. (1989) 'The optic lobe of "
            "Drosophila melanogaster. I. A Golgi analysis of wild-type "
            "structure.' Cell Tissue Res. 258:441-475. doi:10.1007/BF00218858 "
            "-- the Lai/Am nomenclature.",
        ),
        caveat=(
            "The shipped connectome marks Am->L1 and Am->T1 as alpha=+1 with "
            "alpha_fixed=True (source: Nern, personal communication 2018), and "
            "Am->T1 carries 63 of Am's 95 weighted synapses, which is why the "
            "weighted vote returns +1 while the unweighted pair vote returns -1. "
            "Enforcing the published glutamatergic identity therefore FLIPS the "
            "single heaviest Am connection. Report this arm's M3 mean shift, and "
            "run the unambiguous-only ablation (DaleScope='unambiguous_only')."
        ),
        connectome_vote_sign=+1.0,
    ),
    "C2": NeurotransmitterAssignment(
        cell_type="C2",
        aliases=("centrifugal neuron C2",),
        transmitter="GABA",
        receptor="Rdl (GABA-A-type chloride channel)",
        sign=-1.0,
        evidence=(
            "Kolodziejczyk et al. 2008 state verbatim in the abstract that "
            "'GABA is the likely transmitter of centrifugal neurons C2 and C3', "
            "with GABA and GAD immunoreactivity shown in both."
        ),
        references=(
            "Kolodziejczyk A., Sun X., Meinertzhagen I.A., Nassel D.R. (2008) "
            "'Glutamate, GABA and acetylcholine signaling components in the "
            "lamina of the Drosophila visual system.' PLoS ONE 3(5):e2110. "
            "doi:10.1371/journal.pone.0002110",
            "Davis F.P. et al. (2020) 'A genetic, genomic, and computational "
            "resource for exploring neural circuit function.' eLife 9:e50901. "
            "doi:10.7554/eLife.50901 -- optic-lobe cell-type transcriptomes "
            "including transmitter-pathway genes.",
        ),
        caveat=(
            "The connectome marks C2->L1 and C2->L2 as alpha=+1 (Nern, personal "
            "communication 2018). Those two pairs are overridden by the "
            "published GABAergic identity. The weighted vote already favours -1 "
            "(29.95 vs 165.05), so this axis is not a coin flip for C2."
        ),
        connectome_vote_sign=-1.0,
    ),
    "C3": NeurotransmitterAssignment(
        cell_type="C3",
        aliases=("centrifugal neuron C3",),
        transmitter="GABA",
        receptor="Rdl (GABA-A-type chloride channel)",
        sign=-1.0,
        evidence=(
            "Same source and same verbatim sentence as C2: 'GABA is the likely "
            "transmitter of centrifugal neurons C2 and C3' (Kolodziejczyk et al. "
            "2008, abstract), with GABA/GAD immunoreactivity in C3."
        ),
        references=(
            "Kolodziejczyk A., Sun X., Meinertzhagen I.A., Nassel D.R. (2008) "
            "'Glutamate, GABA and acetylcholine signaling components in the "
            "lamina of the Drosophila visual system.' PLoS ONE 3(5):e2110. "
            "doi:10.1371/journal.pone.0002110",
            "Davis F.P. et al. (2020) 'A genetic, genomic, and computational "
            "resource for exploring neural circuit function.' eLife 9:e50901. "
            "doi:10.7554/eLife.50901",
        ),
        caveat=(
            "The connectome marks C3->L1 and C3->L2 as alpha=+1 (Nern, personal "
            "communication 2018); those are overridden. Weighted vote already "
            "favours -1 (62.12 vs 365.58)."
        ),
        connectome_vote_sign=-1.0,
    ),
    "R8": NeurotransmitterAssignment(
        cell_type="R8",
        aliases=("photoreceptor R8",),
        transmitter="histamine (co-transmits acetylcholine)",
        receptor="ort / HisCl1 (histamine-gated chloride channel)",
        sign=-1.0,
        evidence=(
            "Drosophila photoreceptors including R8 are histaminergic (Pollack & "
            "Hofbauer 1991), and histamine opens a chloride conductance in their "
            "postsynaptic partners (Hardie 1989; Gengs et al. 2002; ort is "
            "required in the retinotopic targets, Gao et al. 2008). flyvis's own "
            "JSON cites 'Hardie1989' for exactly the R8 rows it marks alpha=-1 "
            "(R8->R7, L1, L3, Tm5c, Tm20), verified this session."
        ),
        references=(
            "Hardie R.C. (1989) 'A histamine-activated chloride channel involved "
            "in neurotransmission at a photoreceptor synapse.' Nature "
            "339:704-706. doi:10.1038/339704a0",
            "Pollack I. & Hofbauer A. (1991) 'Histamine-like immunoreactivity in "
            "the visual system and brain of Drosophila melanogaster.' Cell Tissue "
            "Res. 266:391-398. doi:10.1007/BF00318195",
            "Gengs C. et al. (2002) 'The target of Drosophila photoreceptor "
            "synaptic transmission is a histamine-gated chloride channel encoded "
            "by ort (hclA).' J. Biol. Chem. 277:42113-42120. "
            "doi:10.1074/jbc.M207133200",
            "Gao S. et al. (2008) 'The neural substrate of spectral preference in "
            "Drosophila.' Neuron 60:328-342. doi:10.1016/j.neuron.2008.08.010",
            "Xiao N. et al. (2023) 'A single photoreceptor splits perception and "
            "entrainment by cotransmission.' Nature 623:562-570. "
            "doi:10.1038/s41586-023-06681-6",
        ),
        caveat=(
            "R8 genuinely violates the one-neuron-one-transmitter idealization: "
            "Xiao et al. 2023 show R8 'separates image-forming and irradiance "
            "signals by co-transmitting two neurotransmitters, histamine and "
            "acetylcholine', and that 'this segregation is further established "
            "postsynaptically by histamine-receptor-expressing unicolumnar "
            "retinotopic neurons and acetylcholine-receptor-expressing "
            "multicolumnar integration neurons' (abstract, quoted verbatim). The "
            "flyvis model contains the unicolumnar retinotopic motion pathway, "
            "so the histaminergic (inhibitory) channel is the correct single "
            "sign for THIS model -- but the assignment is a modelling decision "
            "about which co-transmitter the model represents, not a claim that "
            "R8 has one transmitter. Pre-register it as a sensitivity axis. Note "
            "the connectome's own alpha=+1 R8 rows (R8->Mi1/Mi4/Mi9/Mi15) are "
            "unicolumnar medulla neurons attributed to a personal communication, "
            "not to the co-transmission paper."
        ),
        connectome_vote_sign=-1.0,
    ),
}


def published_explicit_signs(
    restrict_to: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Return ``{cell_type: sign}`` from :data:`PUBLISHED_NT_SIGNS`.

    Args:
        restrict_to: If given, only types present in this collection are
            returned. Pass ``connectome.type_names`` so that a connectome
            lacking one of the four types does not raise.

    Returns:
        Mapping of cell-type name to ``+1.0`` / ``-1.0``.
    """
    names = None if restrict_to is None else set(restrict_to)
    return {
        k: v.sign
        for k, v in PUBLISHED_NT_SIGNS.items()
        if names is None or k in names
    }


def nt_citation_table() -> str:
    """Human-readable table of every published sign and its justification."""
    lines = [
        "Published neurotransmitter -> Dale sign assignments (D6)",
        "Sign convention:",
    ]
    for r in SIGN_CONVENTION_REFERENCES:
        lines.append(f"  - {r}")
    lines.append("")
    for name in AMBIGUOUS_SOURCE_TYPES:
        assignment = PUBLISHED_NT_SIGNS[name]
        lines.append(assignment.summary())
        vote = CONNECTOME_VOTES.get(name)
        if vote:
            lines.append(
                f"  connectome vote: weighted +{vote['weighted_pos']:.2f} / "
                f"-{vote['weighted_neg']:.2f}; pairs +{int(vote['pairs_pos'])} / "
                f"-{int(vote['pairs_neg'])}; flyvis alpha refs "
                f"{', '.join(CONNECTOME_ALPHA_REFERENCES.get(name, ()))}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


# =============================================================================
# D1 -- SHARED SUBSTRATE: sign resolution
# =============================================================================


def _weighted_type_vote(con: "TypedConnectome", pre_type: int) -> Tuple[float, float]:
    """Synapse-count-weighted positive/negative mass emitted by one type.

    Returns ``(pos_mass, neg_mass)`` summed over every edge whose presynaptic
    neuron has type ``pre_type``, weighting each edge by its synapse count.
    """
    if con.edge_sign_hint is None:
        raise ValueError("edge_sign_hint is required to vote on a type's sign")
    mask = con.edge_source_type == pre_type
    if not bool(mask.any()):
        return 0.0, 0.0
    hint = con.edge_sign_hint[mask]
    w = con.syn_count[mask]
    pos = float(w[hint > 0].sum())
    neg = float(w[hint < 0].sum())
    return pos, neg


def mixed_sign_types(con: "TypedConnectome") -> List[str]:
    """Names of presynaptic types that emit BOTH signs in ``con``.

    These are the types on which Dale's law is not already satisfied by the
    connectome, i.e. the types where axis (a) actually changes something. On the
    shipped flyvis connectome this is exactly ``{Am, C2, C3, R8}`` (D6).
    """
    if con.edge_sign_hint is None:
        raise ValueError("edge_sign_hint is required to find mixed-sign types")
    out: List[str] = []
    src_type = con.edge_source_type
    for t in range(con.n_types):
        mask = src_type == t
        if not bool(mask.any()):
            continue
        hint = con.edge_sign_hint[mask]
        if bool((hint > 0).any()) and bool((hint < 0).any()):
            out.append(con.type_names[t])
    return out


def resolve_edge_signs(
    con: "TypedConnectome",
    *,
    sign_rule: SignRule,
    dale_scope: DaleScope = "all",
    sign_policy: SignPolicy = "explicit",
    explicit_signs: Optional[Dict[str, float]] = None,
) -> Tensor:
    """Resolve the FIXED, non-learnable sign of every edge. Treatment axis (a).

    ``sign_rule="connectome_pair"`` reproduces flyvis exactly: the sign is taken
    from the connectome, which is constant within a ``(pre_type, post_type)``
    pair but may vary across the pairs a single type emits.

    ``sign_rule="dale"`` enforces one sign per PRESYNAPTIC TYPE. For the four
    mixed-sign types the sign comes from :data:`PUBLISHED_NT_SIGNS` when
    ``sign_policy="explicit"`` (D6) -- never from a vote, because R8's weighted
    vote is 45.81 vs 42.80, a 3.4% margin, and Am reverses between the weighted
    and unweighted policies. A coin flip there would decide the sign of ~2% of
    type-level synaptic weight across 33 of 65 postsynaptic types, including the
    types the biological-agreement metric scores.

    Args:
        con: The connectome.
        sign_rule: ``"connectome_pair"`` (baseline) or ``"dale"`` (treatment).
        dale_scope: ``"all"`` enforces Dale on every type; ``"unambiguous_only"``
            leaves the mixed-sign types on their per-pair signs. The latter is
            the pre-registered D6 ablation arm.
        sign_policy: How a mixed type's single sign is chosen. ``"explicit"``
            uses published neurotransmitter identity and RAISES if a mixed type
            has no published entry.
        explicit_signs: Override map ``{type_name: +-1.0}``. Defaults to
            :func:`published_explicit_signs` restricted to this connectome.

    Returns:
        ``(n_edges,)`` float tensor of ``+1.0`` / ``-1.0``, non-learnable.
    """
    if con.edge_sign_hint is None:
        raise ValueError("edge_sign_hint is required to resolve signs")
    hint = con.edge_sign_hint.to(torch.float32)
    if sign_rule == "connectome_pair":
        return torch.sign(hint).clamp(min=-1.0, max=1.0).detach().clone()
    if sign_rule != "dale":
        raise ValueError(f"unknown sign_rule {sign_rule!r}")

    if explicit_signs is None:
        explicit_signs = published_explicit_signs(restrict_to=con.type_names)
    mixed = set(mixed_sign_types(con))
    out = torch.sign(hint).clamp(min=-1.0, max=1.0).detach().clone()
    src_type = con.edge_source_type

    for t in range(con.n_types):
        name = con.type_names[t]
        mask = src_type == t
        if not bool(mask.any()):
            continue
        is_mixed = name in mixed
        if is_mixed and dale_scope == "unambiguous_only":
            continue  # pre-registered ablation: leave per-pair signs alone
        if name in explicit_signs:
            sign = float(explicit_signs[name])
        elif not is_mixed:
            sign = float(torch.sign(con.edge_sign_hint[mask][0]))
        elif sign_policy == "explicit":
            raise ValueError(
                f"type {name!r} emits both signs but has no published "
                f"neurotransmitter assignment; add one to PUBLISHED_NT_SIGNS or "
                f"use dale_scope='unambiguous_only'"
            )
        else:
            pos, neg = _weighted_type_vote(con, t)
            if sign_policy == "majority":
                hint_t = con.edge_sign_hint[mask]
                pos, neg = float((hint_t > 0).sum()), float((hint_t < 0).sum())
            sign = 1.0 if pos >= neg else -1.0
        out[mask] = sign
    return out


# =============================================================================
# D1 -- SHARED SUBSTRATE: typed, softplus-parameterized magnitudes
# =============================================================================


def _grouping(con: "TypedConnectome", tying: TyingMode) -> Tuple[Tensor, int, str]:
    """Map each edge to its magnitude-parameter group. Treatment axis (b)."""
    if tying == "per_edge":
        return (
            torch.arange(con.n_edges, device=con.source_index.device),
            con.n_edges,
            "one magnitude per edge",
        )
    if tying == "per_type_pair":
        raw = con.edge_source_type * con.n_types + con.edge_target_type
        uniq, inv = torch.unique(raw, sorted=True, return_inverse=True)
        return inv, int(uniq.numel()), "one magnitude per (pre_type, post_type)"
    if tying == "per_post_type":
        return con.edge_target_type.clone(), con.n_types, "one magnitude per post_type"
    raise ValueError(f"unknown tying mode {tying!r}")


class TypedWeights(nn.Module):
    """Signed, synapse-count-scaled edge weights with tied softplus magnitudes.

    SHARED SUBSTRATE (D1): the magnitude parameterization is ``softplus`` in both
    arms and there is no switch that changes it. ``MAGNITUDE_PARAMETERIZATION``
    is a class attribute precisely so that :func:`assert_shared_substrate` can
    prove the two arms agree on it.

    The effective per-edge weight is::

        w_e = sign_e * softplus(raw[group(e)]) * syn_count_e

    where ``sign_e`` and ``syn_count_e`` are FIXED buffers and ``raw`` is the only
    learnable tensor.
    """

    MAGNITUDE_PARAMETERIZATION = "softplus"

    def __init__(
        self,
        con: "TypedConnectome",
        tying: TyingMode,
        signs: Tensor,
        *,
        magnitude_scale: float = FLYVIS.magnitude_scale,
    ) -> None:
        super().__init__()
        group_index, n_groups, desc = _grouping(con, tying)
        self.tying: TyingMode = tying
        self.n_groups = n_groups
        self.grouping_description = desc
        self.n_edges = con.n_edges

        # flyvis init (SynapseCountScaling): magnitude = rho / <N>, where <N> is the MEAN
        # SYNAPSE COUNT over the edges of the edge's (source_type, target_type) pair -- not
        # the in-degree (audit I23). Computed per pair, then averaged into this tying's
        # groups, so per_edge and per_type_pair start from identical per-edge magnitudes.
        # float64 sums: float32 accumulation over ~2,000 edges per pair drifts ~2e-4.
        count = con.syn_count.to(torch.float64)
        dev = count.device
        _, pair = torch.unique(con.edge_source_type * con.n_types + con.edge_target_type, return_inverse=True)
        n_pairs = int(pair.max()) + 1
        ones = torch.ones_like(count)
        f64 = dict(device=dev, dtype=torch.float64)
        pair_mean = (torch.zeros(n_pairs, **f64).index_add_(0, pair, count)
                     / torch.zeros(n_pairs, **f64).index_add_(0, pair, ones))
        edge_init = float(magnitude_scale) / pair_mean.clamp(min=_EPS)[pair]
        init = (torch.zeros(n_groups, **f64).index_add_(0, group_index, edge_init)
                / torch.zeros(n_groups, **f64).index_add_(0, group_index, ones).clamp(min=1.0))
        self.magnitude_init = float(edge_init.mean())   # edge-weighted; compared across arms
        raw = inverse_softplus(init.to(torch.float32))
        self.raw_magnitude = nn.Parameter(raw)

        self.register_buffer("group_index", group_index)
        self.register_buffer("signs", signs.to(torch.float32))
        self.register_buffer("syn_count", con.syn_count.to(torch.float32))

    def magnitudes(self) -> Tensor:
        """``(n_groups,)`` strictly positive magnitudes."""
        return F.softplus(self.raw_magnitude)

    def semantic_jacobian(self) -> Tensor:
        """d(magnitude)/d(raw) = sigmoid(raw). The science-2 diagnostic.

        Adam's raw step is gradient-scale-invariant, so the SEMANTIC step size is
        ``lr * sigmoid(raw)``. Both arms must share this element-wise at init or
        the comparison is confounded by optimizer geometry rather than biology.
        """
        return torch.sigmoid(self.raw_magnitude)

    def edge_weights(self) -> Tensor:
        """``(n_edges,)`` signed, count-scaled effective weights."""
        return self.signs * self.magnitudes()[self.group_index] * self.syn_count

    def extra_repr(self) -> str:
        return (
            f"tying={self.tying}, n_groups={self.n_groups}, n_edges={self.n_edges}, "
            f"parameterization={self.MAGNITUDE_PARAMETERIZATION}"
        )


# =============================================================================
# D5 / axis (c) -- per-type time constants, static or liquid
# =============================================================================


class PerTypeLiquidTau(nn.Module):
    """One time constant per cell type, optionally input-dependent. Axis (c).

    Static (baseline)::

        tau_i = softplus(raw_tau[type(i)])

    Liquid (treatment), a CfC-style input-dependent time constant::

        tau_i = softplus(raw_tau[type(i)]) * exp(tanh(gv*v_i + gu*u_i + gb))

    with ``gv``, ``gu``, ``gb`` per cell type. This form was chosen over the more
    common ``tau / (1 + softplus(...))`` for three reasons:

    * **Exactly identity at initialization.** All three gates init to 0, so
      ``exp(tanh(0)) == 1`` EXACTLY and the liquid arm's initial time constants
      are bit-identical to the static arm's. A ``1/(1+softplus(z))`` gate is off
      by ``1/(1+log 2)`` at ``z=0`` unless the bias is fudged, which would put a
      silent initialization difference inside the treatment.
    * **Bounded.** ``tau`` stays in ``[tau_base/e, tau_base*e]``, so a diverging
      gate cannot drive ``tau`` to 0 (unstable) or to infinity (frozen).
    * **Strictly positive** for any gate value, with no clamp.

    Parameter count (D7): ``raw_tau(n_types)`` plus ``3 * n_types`` gate entries,
    i.e. 65 static / 260 liquid at ``n_types=65``.
    """

    def __init__(
        self,
        n_types: int,
        *,
        liquid: bool,
        tau_init: float = FLYVIS.tau_init,
        tau_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.liquid = bool(liquid)
        self.n_types = int(n_types)
        self.tau_init = float(tau_init)
        self.tau_min = float(tau_min)
        self.raw_tau = nn.Parameter(
            inverse_softplus(torch.full((n_types,), float(tau_init), dtype=torch.float32))
        )
        if self.liquid:
            self.gate_v = nn.Parameter(torch.zeros(n_types))
            self.gate_u = nn.Parameter(torch.zeros(n_types))
            self.gate_b = nn.Parameter(torch.zeros(n_types))

    def base_tau(self) -> Tensor:
        """``(n_types,)`` static time constants."""
        t = F.softplus(self.raw_tau)
        return t.clamp(min=self.tau_min) if self.tau_min > 0 else t

    def tau(
        self,
        neuron_type: Tensor,
        v: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
    ) -> Tensor:
        """Per-neuron time constant, broadcast from per-type parameters."""
        base = self.base_tau()[neuron_type]
        if not self.liquid:
            return base
        if v is None or u is None:
            return base
        z = (
            self.gate_v[neuron_type] * v
            + self.gate_u[neuron_type] * u
            + self.gate_b[neuron_type]
        )
        return base * torch.exp(torch.tanh(z))

    def n_parameters(self) -> int:
        """Number of learnable entries held by this module."""
        return sum(int(p.numel()) for p in self.parameters())

    def extra_repr(self) -> str:
        return f"n_types={self.n_types}, liquid={self.liquid}, tau_init={self.tau_init}"


# =============================================================================
# arm specification
# =============================================================================


@dataclass(frozen=True)
class ArmSpec:
    """The three treatment axes, and nothing else. See module docstring D1."""

    sign_rule: SignRule
    tying: TyingMode
    liquid_tau: bool
    dale_scope: DaleScope = "all"

    @property
    def axes(self) -> Dict[str, Any]:
        """The axes that :func:`assert_one_variable_apart` compares."""
        return {
            "sign_rule": self.sign_rule,
            "tying": self.tying,
            "liquid_tau": self.liquid_tau,
        }

    def describe(self) -> str:
        return (
            f"sign_rule={self.sign_rule}, tying={self.tying}, "
            f"liquid_tau={self.liquid_tau}, dale_scope={self.dale_scope}"
        )


# =============================================================================
# D5 -- the cell. Exact-exponential integrator in BOTH arms.
# =============================================================================


class DaleTypedCfCCell(nn.Module):
    """One recurrent step of a connectome-constrained network.

    SHARED SUBSTRATE (D5): the integrator is the exact exponential (zero-order
    hold) solution of ``tau dv/dt = -v + u`` in BOTH arms::

        alpha  = exp(-dt / tau)
        v_next = alpha * v + (1 - alpha) * u

    flyvis instead uses explicit forward Euler with a ``1/max(tau, dt)`` guard.
    At flyvis's own operating point ``dt/tau = 0.02/0.05 = 0.4``, per-step
    retention is 0.600 for Euler against 0.670 for the exact exponential -- an
    11.7% relative difference. Leaving Euler in the baseline while the treatment
    arm got a continuous-time update would have made the integrator a second
    treatment variable, so both arms get the exact form and the ONLY difference
    is whether ``tau`` is input-dependent.

    The state is per-neuron voltage. Nothing here trains, saves or plots.
    """

    INTEGRATOR = "exact_exponential"

    def __init__(
        self,
        con: "TypedConnectome",
        weights: TypedWeights,
        tau: PerTypeLiquidTau,
        spec: ArmSpec,
        *,
        dt: float = FLYVIS.dt,
        bias_mean: float = FLYVIS.bias_mean,
        bias_std: float = FLYVIS.bias_std,
        activation: Callable[[Tensor], Tensor] = F.relu,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.tau_module = tau
        self.spec = spec
        self.dt = float(dt)
        self.activation = activation
        self.n_neurons = con.n_neurons
        self.n_types = con.n_types
        self.type_names = list(con.type_names)

        # Draw on CPU, where build_arm's generator lives, then move. `import flyvis` makes
        # CUDA the default device on a GPU runtime; drawing there with a CPU generator
        # raises (audit K3). Same numbers on every device for a given seed.
        bias = torch.empty(con.n_types, dtype=torch.float32, device="cpu")
        bias.normal_(mean=bias_mean, std=bias_std, generator=generator)
        self.bias = nn.Parameter(bias.to(con.source_index.device))

        self.register_buffer("source_index", con.source_index.clone())
        self.register_buffer("target_index", con.target_index.clone())
        self.register_buffer("neuron_type", con.neuron_type.clone())

    # -- dynamics -------------------------------------------------------------

    def input_current(self, v: Tensor, x_ext: Optional[Tensor] = None) -> Tensor:
        """Total incoming current per neuron: recurrent + bias + external."""
        a = self.activation(v)
        w = self.weights.edge_weights()
        contrib = w * a.index_select(-1, self.source_index)
        u = torch.zeros_like(v)
        u.index_add_(-1, self.target_index, contrib)
        u = u + self.bias[self.neuron_type]
        if x_ext is not None:
            u = u + x_ext
        return u

    def retention(self, tau: Tensor) -> Tensor:
        """``alpha = exp(-dt / tau)``, the exact-exponential retention factor."""
        return torch.exp(-self.dt / tau.clamp(min=_EPS))

    def forward(self, v: Tensor, x_ext: Optional[Tensor] = None) -> Tensor:
        """Advance the state one timestep of length ``dt``."""
        u = self.input_current(v, x_ext)
        tau = self.tau_module.tau(self.neuron_type, v=v, u=u)
        alpha = self.retention(tau)
        return alpha * v + (1.0 - alpha) * u

    def euler_retention_reference(self) -> Tensor:
        """flyvis's Euler retention, for the D5 comparison only. Not used."""
        base = self.tau_module.base_tau()[self.neuron_type]
        return 1.0 - self.dt / torch.clamp(base, min=self.dt)

    # -- reporting ------------------------------------------------------------

    def parameter_report(self) -> Dict[str, Any]:
        """Authoritative learnable-parameter accounting (D7).

        The round-1 review found the parameter count mis-stated by 3-6x because
        it was computed by hand. This method is the single source of truth; the
        smoke test and the docs both read it rather than re-deriving it.
        """
        n_mag = int(self.weights.raw_magnitude.numel())
        n_tau = self.tau_module.n_parameters()
        n_bias = int(self.bias.numel())
        total = n_mag + n_tau + n_bias
        baseline_total = n_mag + self.n_types + n_bias  # static tau
        return {
            "magnitude": n_mag,
            "tau": n_tau,
            "bias": n_bias,
            "total": total,
            "baseline_equivalent_total": baseline_total,
            "delta_vs_baseline": total - baseline_total,
            "pct_vs_baseline": 100.0 * (total - baseline_total) / max(baseline_total, 1),
            "n_edges": int(self.weights.n_edges),
            "tying": self.weights.tying,
            "reduction_factor": self.weights.n_edges / max(n_mag, 1),
            "liquid_tau": self.tau_module.liquid,
            "integrator": self.INTEGRATOR,
            "magnitude_parameterization": self.weights.MAGNITUDE_PARAMETERIZATION,
        }

    def parameter_matched_filler(self, reference_total: Optional[int] = None) -> int:
        """Extra params a parameter-matched control arm needs (science-4).

        The killer confounder for the headline result is that the constrained arm
        might simply have fewer parameters, hence less room to vary. The control
        is a baseline padded to the SAME parameter count. This returns the pad
        size, e.g. 929 - 734 = 195 on the real flyvis shape.
        """
        rep = self.parameter_report()
        ref = (
            reference_total
            if reference_total is not None
            else rep["baseline_equivalent_total"]
        )
        return max(int(rep["total"]) - int(ref), 0)

    def extra_repr(self) -> str:
        return f"n_neurons={self.n_neurons}, dt={self.dt}, {self.spec.describe()}"


# =============================================================================
# D1 -- the single constructor, and the proofs that the arms are comparable
# =============================================================================


def build_arm(
    con: "TypedConnectome",
    *,
    dale: bool,
    liquid_tau: bool,
    tying: TyingMode = "per_type_pair",
    dale_scope: DaleScope = "all",
    sign_policy: SignPolicy = "explicit",
    explicit_signs: Optional[Dict[str, float]] = None,
    dt: float = FLYVIS.dt,
    tau_init: float = FLYVIS.tau_init,
    tau_min: float = 0.0,
    magnitude_scale: float = FLYVIS.magnitude_scale,
    bias_mean: float = FLYVIS.bias_mean,
    bias_std: float = FLYVIS.bias_std,
    activation: Callable[[Tensor], Tensor] = F.relu,
    seed: int = 0,
) -> DaleTypedCfCCell:
    """Build one experimental arm. THE ONLY WAY to construct a model here.

    Both arms come out of this function, so the shared substrate cannot be given
    to one arm and withheld from the other -- softplus magnitudes (D1) and the
    exact-exponential integrator (D5) are structural, not optional.

    Example::

        baseline    = build_arm(con, dale=False, liquid_tau=False)
        constrained = build_arm(con, dale=True,  liquid_tau=True)
        assert_shared_substrate(baseline, constrained)

    Args:
        con: The connectome. Frozen; no edges are added or removed.
        dale: Axis (a). ``False`` = flyvis's per-pair sign rule; ``True`` = one
            fixed sign per presynaptic cell type.
        liquid_tau: Axis (c). Whether per-type time constants are input-dependent.
        tying: Axis (b). Magnitude tying granularity.
        dale_scope: ``"unambiguous_only"`` selects the pre-registered D6 ablation.
        sign_policy: How mixed-sign types get their single sign. ``"explicit"``
            (default) requires a published neurotransmitter assignment.
        seed: Seeds the bias draw only. Everything else is deterministic.

    Returns:
        A :class:`DaleTypedCfCCell`.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sign_rule: SignRule = "dale" if dale else "connectome_pair"
    signs = resolve_edge_signs(
        con,
        sign_rule=sign_rule,
        dale_scope=dale_scope,
        sign_policy=sign_policy,
        explicit_signs=explicit_signs,
    )
    weights = TypedWeights(con, tying, signs, magnitude_scale=magnitude_scale)
    tau = PerTypeLiquidTau(
        con.n_types, liquid=liquid_tau, tau_init=tau_init, tau_min=tau_min
    )
    spec = ArmSpec(
        sign_rule=sign_rule,
        tying=tying,
        liquid_tau=bool(liquid_tau),
        dale_scope=dale_scope,
    )
    return DaleTypedCfCCell(
        con,
        weights,
        tau,
        spec,
        dt=dt,
        bias_mean=bias_mean,
        bias_std=bias_std,
        activation=activation,
        generator=generator,
    )


def assert_one_variable_apart(a: DaleTypedCfCCell, b: DaleTypedCfCCell) -> str:
    """Raise unless ``a`` and ``b`` differ in AT MOST ONE treatment axis."""
    diffs = [k for k, v in a.spec.axes.items() if b.spec.axes[k] != v]
    if len(diffs) > 1:
        raise AssertionError(
            f"arms differ in {len(diffs)} axes {diffs}; a one-variable comparison "
            f"requires at most one. a: {a.spec.describe()} | b: {b.spec.describe()}"
        )
    return diffs[0] if diffs else "(identical)"


def differing_axes(a: DaleTypedCfCCell, b: DaleTypedCfCCell) -> List[str]:
    """Names of the treatment axes on which two arms differ."""
    return [k for k, v in a.spec.axes.items() if b.spec.axes[k] != v]


def assert_shared_substrate(
    a: DaleTypedCfCCell,
    b: DaleTypedCfCCell,
    *,
    atol: float = 0.0,
    require_one_variable: bool = True,
) -> Dict[str, Any]:
    """Prove, tensor by tensor, that two arms share everything but the axes.

    This is the machine-checkable form of D1. It is what stops the killer
    objection (science-1) from creeping back in through a refactor: if anyone
    ever gives softplus, the integrator, the initialization or the fixed synapse
    counts to one arm only, this raises.

    Args:
        a: One arm.
        b: The other arm.
        atol: Absolute tolerance for the tensor comparisons. Defaults to exact.
        require_one_variable: If True (default) also require that the arms differ
            in at most one treatment axis. Set False to check the substrate of a
            MULTI-axis pair, e.g. the fully constrained arm against the baseline.

            Note that the fully constrained arm (``dale=True, liquid_tau=True``)
            differs from the baseline in TWO axes, so a single
            treatment-vs-baseline contrast is NOT one-variable. The design
            therefore needs a factorial ladder -- baseline, Dale-only,
            liquid-only, both -- and each ADJACENT pair in that lattice is what
            this function validates with the default setting. Attributing a
            combined effect to one axis without the single-axis arms is exactly
            the confound the round-1 review flagged.

    Returns a dict of the checks performed, for logging into the run record.
    """
    checks: Dict[str, Any] = {}

    def _same(name: str, x: Any, y: Any) -> None:
        if x != y:
            raise AssertionError(f"shared substrate violated: {name}: {x!r} != {y!r}")
        checks[name] = x

    _same("integrator", a.INTEGRATOR, b.INTEGRATOR)
    _same(
        "magnitude_parameterization",
        a.weights.MAGNITUDE_PARAMETERIZATION,
        b.weights.MAGNITUDE_PARAMETERIZATION,
    )
    _same("dt", a.dt, b.dt)
    _same("activation", a.activation, b.activation)
    _same("n_neurons", a.n_neurons, b.n_neurons)
    _same("n_types", a.n_types, b.n_types)
    _same("magnitude_init", a.weights.magnitude_init, b.weights.magnitude_init)
    _same("tau_init", a.tau_module.tau_init, b.tau_module.tau_init)

    def _close(name: str, x: Tensor, y: Tensor) -> None:
        if x.shape != y.shape:
            raise AssertionError(
                f"shared substrate violated: {name}: shape {tuple(x.shape)} != {tuple(y.shape)}"
            )
        if not torch.allclose(x, y, atol=atol, rtol=0.0):
            worst = float((x - y).abs().max())
            raise AssertionError(
                f"shared substrate violated: {name}: max|diff| = {worst:g}"
            )
        checks[name] = "equal"

    _close("syn_count", a.weights.syn_count, b.weights.syn_count)
    _close("bias", a.bias.detach(), b.bias.detach())
    _close("base_tau", a.tau_module.base_tau().detach(), b.tau_module.base_tau().detach())

    # The science-2 guarantee: identical semantic step size at initialization.
    if a.weights.tying == b.weights.tying:
        _close(
            "magnitudes",
            a.weights.magnitudes().detach(),
            b.weights.magnitudes().detach(),
        )
        _close(
            "semantic_jacobian",
            a.weights.semantic_jacobian().detach(),
            b.weights.semantic_jacobian().detach(),
        )

    # Dale may only change signs, never their magnitude or support.
    _close(
        "abs_edge_weights",
        a.weights.edge_weights().detach().abs(),
        b.weights.edge_weights().detach().abs(),
    )

    # ArmSpec fields are LABELS. Prove each declared treatment difference (and sameness)
    # is real in the tensors, so a no-op or leaked treatment cannot pass as a contrast
    # (audit I9). Note: on the fib25/fib19 connectome dale_scope="unambiguous_only" flips
    # nothing, so arm C vs arm A is refused here by design.
    n_flips = int((a.weights.signs != b.weights.signs).sum())
    same_sign_spec = (a.spec.sign_rule, a.spec.dale_scope) == (b.spec.sign_rule, b.spec.dale_scope)
    if same_sign_spec != (n_flips == 0):
        raise AssertionError(
            f"treatment not applied: sign spec {a.spec.sign_rule}/{a.spec.dale_scope} vs "
            f"{b.spec.sign_rule}/{b.spec.dale_scope} but {n_flips} edge signs differ")
    checks["n_sign_flips"] = n_flips
    for arm in (a, b):
        gated = all(isinstance(getattr(arm.tau_module, g, None), nn.Parameter)
                    for g in ("gate_v", "gate_u", "gate_b"))
        if not (arm.spec.liquid_tau == arm.tau_module.liquid == gated):
            raise AssertionError(
                f"treatment not applied: spec.liquid_tau={arm.spec.liquid_tau} but "
                f"tau_module.liquid={arm.tau_module.liquid}, gates present={gated}")

    checks["differing_axes"] = differing_axes(a, b)
    if require_one_variable:
        checks["differing_axis"] = assert_one_variable_apart(a, b)
    return checks


# =============================================================================
# the factorial ladder the one-variable requirement implies
# =============================================================================


def factorial_ladder(
    con: "TypedConnectome", *, tying: TyingMode = "per_type_pair", seed: int = 0, **kw: Any
) -> Dict[str, DaleTypedCfCCell]:
    """The four arms needed to attribute an effect to a single axis.

    Because the fully constrained arm differs from the baseline in two axes
    (Dale and liquid tau), a two-arm design cannot say which axis produced any
    change in mechanism variance. This builds the 2x2::

        baseline      dale=False liquid=False   (flyvis-equivalent)
        dale_only     dale=True  liquid=False
        liquid_only   dale=False liquid=True
        both          dale=True  liquid=True

    Each arm shares the substrate with every other (softplus magnitudes, exact
    exponential integrator, identical init), and each ADJACENT pair differs in
    exactly one axis, so ``baseline -> dale_only``, ``baseline -> liquid_only``,
    ``dale_only -> both`` and ``liquid_only -> both`` are all one-variable
    contrasts. Interaction is then ``both - dale_only - liquid_only + baseline``.

    Budget consequence: four arms, not two. Cost this honestly rather than
    running two arms and attributing the result to whichever axis is the story.
    """
    combos = {
        "baseline": (False, False),
        "dale_only": (True, False),
        "liquid_only": (False, True),
        "both": (True, True),
    }
    return {
        name: build_arm(con, dale=d, liquid_tau=l, tying=tying, seed=seed, **kw)
        for name, (d, l) in combos.items()
    }


# =============================================================================
# D7 -- ParameterReport: the single source of the parameter arithmetic
# =============================================================================


@dataclass(frozen=True)
class ParameterReport:
    """Learnable-parameter accounting, checkable against a live module.

    The round-1 review found the parameter count mis-stated by 3-6x because it
    was derived by hand in prose. This dataclass exists so the arithmetic has
    exactly one home, and :meth:`check_against_module` makes it impossible for
    the documented number and the real module to drift apart again.
    """

    magnitude: int
    tau: int
    bias: int

    @property
    def total(self) -> int:
        """Total learnable scalars."""
        return self.magnitude + self.tau + self.bias

    @property
    def baseline_equivalent(self) -> int:
        """The same model with a STATIC per-type tau (flyvis's count)."""
        n_types = self.bias  # one bias per cell type
        return self.magnitude + n_types + self.bias

    @property
    def delta(self) -> int:
        """Extra parameters the liquid tau costs; the filler size for the control."""
        return self.total - self.baseline_equivalent

    @property
    def pct(self) -> float:
        """Percent increase over the baseline-equivalent count."""
        return 100.0 * self.delta / max(self.baseline_equivalent, 1)

    @classmethod
    def for_fly_connectome(cls, *, liquid_tau: bool = True) -> "ParameterReport":
        """The real flyvis shape, from :data:`FLYVIS` rather than by hand.

        Gives 929 (+26.6%, filler 195) for ``liquid_tau=True`` and exactly
        flyvis's 734 for ``liquid_tau=False``.
        """
        n_types = FLYVIS.n_types
        return cls(
            magnitude=FLYVIS.n_type_pairs_checkpoint,
            tau=(4 * n_types) if liquid_tau else n_types,
            bias=n_types,
        )

    @classmethod
    def from_module(cls, cell: "DaleTypedCfCCell") -> "ParameterReport":
        """Read the report off a live cell."""
        return cls(
            magnitude=int(cell.weights.raw_magnitude.numel()),
            tau=cell.tau_module.n_parameters(),
            bias=int(cell.bias.numel()),
        )

    def check_against_module(self, cell: "DaleTypedCfCCell") -> None:
        """Raise unless this report equals the module's real parameter count."""
        live = sum(int(p.numel()) for p in cell.parameters())
        mine = ParameterReport.from_module(cell)
        if mine != self:
            raise AssertionError(f"report mismatch: {self} != module's {mine}")
        if live != self.total:
            raise AssertionError(
                f"parameter count drift: report says {self.total}, "
                f"sum(p.numel()) says {live}"
            )

    def describe(self) -> str:
        return (
            f"magnitude {self.magnitude} + tau {self.tau} + bias {self.bias} "
            f"= {self.total}  (baseline-equivalent {self.baseline_equivalent}, "
            f"+{self.delta} = +{self.pct:.1f}%)"
        )


# =============================================================================
# D5 -- integrator comparison and the learned-tau-vs-dt distribution
# =============================================================================


def exact_retention(dt: float, tau: Tensor | float) -> Tensor:
    """``exp(-dt/tau)`` -- the exact zero-order-hold retention factor."""
    t = tau if isinstance(tau, Tensor) else torch.tensor(float(tau))
    return torch.exp(-float(dt) / t.clamp(min=_EPS))


def flyvis_euler_retention(dt: float, tau: Tensor | float) -> Tensor:
    """``1 - dt/max(tau, dt)`` -- flyvis's forward-Euler retention, for reference.

    Reproduces ``flyvis/network/dynamics.py`` (velocity scaled by
    ``1/max(time_const, dt)``) composed with ``state + vel*dt`` in
    ``flyvis/network/network.py``. Not used by either arm; kept so the D5
    difference can be quoted rather than asserted.
    """
    t = tau if isinstance(tau, Tensor) else torch.tensor(float(tau))
    return 1.0 - float(dt) / torch.clamp(t, min=float(dt))


@dataclass(frozen=True)
class IntegratorRow:
    """One row of the D5 comparison table."""

    tau: float
    dt: float
    euler: float
    exact: float

    @property
    def rel_diff(self) -> float:
        """Relative difference, exact vs Euler, per step."""
        return abs(self.exact - self.euler) / max(abs(self.euler), _EPS)

    @property
    def euler_gradient_is_dead(self) -> bool:
        """True where tau < dt, so flyvis's guard pins tau and d/dtau is zero."""
        return self.tau < self.dt


def integrator_comparison(
    dt: float = FLYVIS.dt, taus: Optional[Sequence[float]] = None
) -> List[IntegratorRow]:
    """Recompute the D5 table in the module docstring.

    At flyvis's operating point (``dt=0.02``, ``tau=0.05``) this yields
    Euler 0.600000 against exact 0.670320, an 11.72% per-step difference that
    compounds over the rollout.
    """
    if taus is None:
        taus = [0.005, 0.01, FLYVIS.dt, 0.03, FLYVIS.tau_init, 0.1, 0.5]
    rows: List[IntegratorRow] = []
    for tau in taus:
        rows.append(
            IntegratorRow(
                tau=float(tau),
                dt=float(dt),
                euler=float(flyvis_euler_retention(dt, tau)),
                exact=float(exact_retention(dt, tau)),
            )
        )
    return rows


@dataclass(frozen=True)
class TauDtReport:
    """Where an arm's learned time constants sit relative to ``dt``."""

    dt: float
    n_types: int
    n_below_dt: int
    min_tau: float
    median_tau: float
    max_tau: float

    @property
    def frac_below_dt(self) -> float:
        """Fraction of cell types whose tau is below dt.

        In the released ensemble the round-1 reviewer measured 73.7% of
        (model, cell-type) time constants below dt -- a regime where flyvis's
        Euler guard zeroes the tau gradient entirely. The exact-exponential
        update has no such regime, so this is reported per arm rather than
        assumed away.
        """
        return self.n_below_dt / max(self.n_types, 1)

    def describe(self) -> str:
        return (
            f"tau vs dt={self.dt}: {self.n_below_dt}/{self.n_types} types below dt "
            f"({100*self.frac_below_dt:.1f}%); min {self.min_tau:.4g}, "
            f"median {self.median_tau:.4g}, max {self.max_tau:.4g}"
        )


def tau_dt_report(cell: "DaleTypedCfCCell") -> TauDtReport:
    """Summarize an arm's base time constants against its timestep."""
    with torch.no_grad():
        tau = cell.tau_module.base_tau().detach().flatten()
    return TauDtReport(
        dt=cell.dt,
        n_types=int(tau.numel()),
        n_below_dt=int((tau < cell.dt).sum()),
        min_tau=float(tau.min()),
        median_tau=float(tau.median()),
        max_tau=float(tau.max()),
    )


# =============================================================================
# pre-registered arm definitions (D1)
# =============================================================================

#: The treatment axes, in the order they are reported. See :class:`ArmSpec`.
ARM_AXES: Tuple[str, ...] = ("sign_rule", "tying", "liquid_tau")

#: Arm A. flyvis's sign rule and a static tau, on the shared substrate.
BASELINE_ARM = ArmSpec(sign_rule="connectome_pair", tying="per_type_pair", liquid_tau=False)

#: Arm B. Dale's law plus input-dependent per-type time constants.
CONSTRAINED_ARM = ArmSpec(sign_rule="dale", tying="per_type_pair", liquid_tau=True)

#: The pre-registered arms, by name. NOTE that BASELINE_ARM and CONSTRAINED_ARM
#: differ in TWO axes, so that contrast alone cannot attribute an effect to
#: either one; :func:`factorial_ladder` builds the 2x2 that can. PLAN.md prices a
#: fully powered factorial at 4x the seeds and funds one ablation arm instead.
PREREGISTERED_ARMS: Dict[str, ArmSpec] = {
    "baseline": BASELINE_ARM,
    "constrained": CONSTRAINED_ARM,
    "dale_only": ArmSpec(sign_rule="dale", tying="per_type_pair", liquid_tau=False),
    "liquid_only": ArmSpec(sign_rule="connectome_pair", tying="per_type_pair", liquid_tau=True),
    "dale_unambiguous_only": ArmSpec(
        sign_rule="dale", tying="per_type_pair", liquid_tau=True, dale_scope="unambiguous_only"
    ),
}


# =============================================================================
# NOT IMPLEMENTED -- declared so nobody assumes otherwise
# =============================================================================

#: Names the original draft of this module exported but which do NOT exist here.
#: They were lost when a background repair pass was killed mid-rewrite, and are
#: recorded rather than silently dropped. Nothing in Stage 0 needs them; the
#: flyvis seam is required for Stage 2 and is the substantive gap.
#:
#:   FLYVIS SEAM -- implemented in src/flyvis_seam.py (see verify_seam there):
#:     register_flyvis_adapters, flyvis_arm_config, assert_configs_share_substrate
#:
#:   SCIENCE-2 GEOMETRY DIAGNOSTICS (pre-registered in PREREGISTRATION.md):
#:     semantic_snapshot, semantic_displacement, SemanticSnapshot,
#:     SemanticDisplacement, mechanism_fingerprint, mechanism_distance,
#:     mechanism_variance, ArmGeometryReport, arm_geometry_report,
#:     compare_arm_geometry
#:       -- partially available elsewhere: TypedWeights.semantic_jacobian()
#:          provides the init-time check that the arms share Adam geometry, and
#:          src/stability.py implements the mechanism metrics and dispersion
#:          tests. The per-arm displacement reporting is genuinely absent.
#:
#:   REPORT WRAPPERS (cosmetic):
#:     DaleSignResolver, DaleSignReport, SharedSubstrateReport
#:       -- superseded by resolve_edge_signs(), mixed_sign_types() and the dict
#:          returned by assert_shared_substrate().
NOT_IMPLEMENTED: Dict[str, str] = {
    "register_flyvis_adapters": "MOVED: now in src/flyvis_seam.py (not this module)",
    "flyvis_arm_config": "MOVED: now in src/flyvis_seam.py (not this module)",
    "assert_configs_share_substrate": "MOVED: now in src/flyvis_seam.py (not this module)",
    "semantic_snapshot": "science-2 diagnostic; partial via TypedWeights.semantic_jacobian",
    "semantic_displacement": "science-2 diagnostic; absent",
    "mechanism_fingerprint": "see src/stability.py",
    "mechanism_distance": "see src/stability.py",
    "mechanism_variance": "see src/stability.py",
    "arm_geometry_report": "science-2 diagnostic; absent",
    "compare_arm_geometry": "science-2 diagnostic; absent",
    "DaleSignResolver": "superseded by resolve_edge_signs()",
    "DaleSignReport": "superseded by resolve_edge_signs()",
    "SharedSubstrateReport": "superseded by assert_shared_substrate()'s dict",
}


def missing_api() -> str:
    """Human-readable list of the declared-but-absent API. Call it before Stage 2."""
    lines = ["Declared in an earlier draft but NOT implemented here:"]
    for name, why in sorted(NOT_IMPLEMENTED.items()):
        lines.append(f"  {name:<32} {why}")
    lines.append("")
    lines.append("Stage 0 needs none of these. The flyvis seam lives in src/flyvis_seam.py.")
    return "\n".join(lines)


# =============================================================================
# smoke test -- CPU only, no flyvis, no CUDA, no network. Run: python this_file
# =============================================================================


def _synthetic_connectome(make_l1_mixed: bool = False) -> "TypedConnectome":
    """A tiny hand-built connectome with the structural features that matter.

    Five cell types of four neurons each. Signs are constant within every
    ``(pre_type, post_type)`` pair, as in flyvis. ``Am`` and ``R8`` each emit
    BOTH signs, so they are mixed-sign types with published assignments (D6).

    Args:
        make_l1_mixed: If True, flip one ``L1`` edge so ``L1`` also becomes
            mixed. ``L1`` has no published neurotransmitter entry, which is how
            the test exercises the ``sign_policy="explicit"`` failure path.
    """
    type_names = ["Am", "R8", "L1", "Mi1", "T4c"]
    neuron_type = torch.arange(20) // 4  # 4 neurons per type

    #        src, tgt, sign, n_syn
    edges = [
        (0, 8, -1.0, 12.0),   # Am  -> L1   (negative)
        (1, 12, 1.0, 7.0),    # Am  -> Mi1  (positive)  => Am is mixed
        (4, 9, 1.0, 20.0),    # R8  -> L1   (positive)
        (5, 13, -1.0, 3.0),   # R8  -> Mi1  (negative)  => R8 is mixed
        (8, 14, -1.0, 9.0),   # L1  -> Mi1
        (9, 16, -1.0, 15.0),  # L1  -> T4c
        (12, 17, 1.0, 11.0),  # Mi1 -> T4c
        (13, 10, 1.0, 4.0),   # Mi1 -> L1
        (16, 11, 1.0, 6.0),   # T4c -> L1
        (17, 15, 1.0, 8.0),   # T4c -> Mi1
    ]
    if make_l1_mixed:
        edges[5] = (9, 16, 1.0, 15.0)  # flip L1 -> T4c positive

    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    tgt = torch.tensor([e[1] for e in edges], dtype=torch.long)
    sign = torch.tensor([e[2] for e in edges], dtype=torch.float32)
    nsyn = torch.tensor([e[3] for e in edges], dtype=torch.float32)
    return TypedConnectome(
        source_index=src,
        target_index=tgt,
        neuron_type=neuron_type,
        type_names=type_names,
        syn_count=nsyn,
        edge_sign_hint=sign,
    )


def _main() -> int:
    import time as _time

    t0 = _time.perf_counter()
    checks = 0
    line = "=" * 78

    def ok(msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  [PASS] {msg}")

    print(line)
    print("dale_typed_cfc.py self-test")
    print(line)

    con = _synthetic_connectome()
    assert con.n_edges == 10 and con.n_neurons == 20 and con.n_types == 5
    ok(f"synthetic connectome: {con.n_neurons} neurons, {con.n_edges} edges, {con.n_types} types")

    # -- D6: mixed-sign detection and published signs ------------------------
    print("\n-- D6: Dale signs from published neurotransmitter identity --")
    mixed = mixed_sign_types(con)
    assert sorted(mixed) == ["Am", "R8"], mixed
    ok(f"mixed-sign presynaptic types detected: {sorted(mixed)}")

    pub = published_explicit_signs(restrict_to=con.type_names)
    assert set(pub) >= {"Am", "R8"}, pub
    assert all(v in (-1.0, 1.0) for v in pub.values())
    ok(f"published signs available for {sorted(pub)} -> {pub}")

    signs_dale = resolve_edge_signs(con, sign_rule="dale")
    src_type = con.edge_source_type
    for t in range(con.n_types):
        m = src_type == t
        if bool(m.any()):
            u = torch.unique(signs_dale[m])
            assert u.numel() == 1, f"type {con.type_names[t]} emits {u.tolist()}"
    ok("Dale enforced: every presynaptic type emits exactly one sign")

    for name, want in pub.items():
        m = src_type == con.type_id(name)
        got = torch.unique(signs_dale[m]).tolist()
        assert got == [want], f"{name}: got {got}, want [{want}]"
    ok("mixed types took their PUBLISHED sign, not a vote")

    signs_pair = resolve_edge_signs(con, sign_rule="connectome_pair")
    assert torch.equal(signs_pair, torch.sign(con.edge_sign_hint))
    n_flipped = int((signs_dale != signs_pair).sum())
    ok(f"baseline sign rule reproduces the connectome; Dale flips {n_flipped}/{con.n_edges} edges")

    # -- D6 ablation and the explicit-policy guard ---------------------------
    signs_ablate = resolve_edge_signs(con, sign_rule="dale", dale_scope="unambiguous_only")
    for name in ("Am", "R8"):
        m = src_type == con.type_id(name)
        assert torch.equal(signs_ablate[m], signs_pair[m])
    ok("dale_scope='unambiguous_only' leaves the mixed types on per-pair signs")

    con_l1 = _synthetic_connectome(make_l1_mixed=True)
    assert "L1" in mixed_sign_types(con_l1)
    try:
        resolve_edge_signs(con_l1, sign_rule="dale", sign_policy="explicit")
    except ValueError as exc:
        assert "L1" in str(exc)
        ok("explicit policy REFUSES a mixed type with no published assignment")
    else:
        raise AssertionError("expected ValueError for unpublished mixed type L1")

    voted = resolve_edge_signs(con_l1, sign_rule="dale", sign_policy="weighted_majority")
    assert torch.unique(voted[con_l1.edge_source_type == con_l1.type_id("L1")]).numel() == 1
    ok("weighted_majority fallback still resolves a single sign when asked explicitly")

    # -- D1: shared substrate -------------------------------------------------
    print("\n-- D1: shared substrate (the killer-objection fix) --")
    baseline = build_arm(con, dale=False, liquid_tau=False)
    dale_only = build_arm(con, dale=True, liquid_tau=False)
    liquid_only = build_arm(con, dale=False, liquid_tau=True)
    both = build_arm(con, dale=True, liquid_tau=True)

    for m in (baseline, dale_only, liquid_only, both):
        assert m.weights.MAGNITUDE_PARAMETERIZATION == "softplus"
        assert m.INTEGRATOR == "exact_exponential"
    ok("all four arms: softplus magnitudes + exact-exponential integrator")

    for m in (baseline, dale_only, liquid_only, both):
        mags = m.weights.magnitudes()
        assert bool((mags > 0).all()), "softplus produced a non-positive magnitude"
        assert int((mags == 0).sum()) == 0
    ok("no magnitude is exactly zero in ANY arm -- the 418/604 clamp point mass is gone by construction")

    c1 = assert_shared_substrate(baseline, dale_only)
    assert c1["differing_axis"] == "sign_rule", c1
    ok(f"baseline vs dale_only: substrate identical, single differing axis = {c1['differing_axis']}")

    c2 = assert_shared_substrate(baseline, liquid_only)
    assert c2["differing_axis"] == "liquid_tau", c2
    ok(f"baseline vs liquid_only: substrate identical, single differing axis = {c2['differing_axis']}")

    c3 = assert_shared_substrate(baseline, both, require_one_variable=False)
    assert sorted(c3["differing_axes"]) == ["liquid_tau", "sign_rule"], c3
    ok("baseline vs both: substrate identical, 2 differing axes (reported, not hidden)")

    try:
        assert_shared_substrate(baseline, both)
    except AssertionError as exc:
        assert "2 axes" in str(exc)
        ok("the fully constrained arm is REFUSED as a one-variable contrast (factorial ladder required)")
    else:
        raise AssertionError("expected a refusal for the 2-axis contrast")

    ladder = factorial_ladder(con)
    assert set(ladder) == {"baseline", "dale_only", "liquid_only", "both"}
    for a_name, b_name in (
        ("baseline", "dale_only"),
        ("baseline", "liquid_only"),
        ("dale_only", "both"),
        ("liquid_only", "both"),
    ):
        assert_one_variable_apart(ladder[a_name], ladder[b_name])
    ok("factorial ladder: all 4 adjacent pairs are one-variable contrasts")

    # -- science-2: identical optimizer geometry -----------------------------
    jb = baseline.weights.semantic_jacobian().detach()
    jd = dale_only.weights.semantic_jacobian().detach()
    assert torch.equal(jb, jd)
    ok(f"semantic Jacobian sigmoid(raw) identical across arms (value {float(jb[0]):.6g}) -- no Adam-geometry confound")

    # -- D5: exact exponential vs flyvis Euler -------------------------------
    print("\n-- D5: integrator --")
    # .detach() before float(): tau/retention are derived from learnable parameters, so a
    # bare float() emits "Converting a tensor with requires_grad=True to a scalar".
    tau_n = baseline.tau_module.tau(baseline.neuron_type).detach()
    alpha_exact = float(baseline.retention(tau_n).detach()[0])
    alpha_euler = float(baseline.euler_retention_reference().detach()[0])
    want_exact = math.exp(-FLYVIS.dt / FLYVIS.tau_init)
    want_euler = 1.0 - FLYVIS.dt / FLYVIS.tau_init
    assert abs(alpha_exact - want_exact) < 1e-6, (alpha_exact, want_exact)
    assert abs(alpha_euler - want_euler) < 1e-6, (alpha_euler, want_euler)
    rel = abs(alpha_exact - alpha_euler) / alpha_euler
    assert 0.11 < rel < 0.13, rel
    ok(f"retention: exact {alpha_exact:.4f} vs flyvis Euler {alpha_euler:.4f} -> {rel:.1%} apart (both arms use exact)")

    # -- axis (c): tau identical at init, input-dependent after --------------
    t_static = baseline.tau_module.tau(baseline.neuron_type)
    g_dev = torch.get_default_device()
    v = torch.randn(con.n_neurons, generator=torch.Generator(device=g_dev).manual_seed(1))
    u = torch.randn(con.n_neurons, generator=torch.Generator(device=g_dev).manual_seed(2))
    t_liquid = liquid_only.tau_module.tau(liquid_only.neuron_type, v=v, u=u)
    assert torch.equal(t_static, baseline.tau_module.tau(baseline.neuron_type, v=v, u=u))
    assert torch.allclose(t_liquid, t_static, atol=0.0, rtol=0.0)
    ok("liquid tau is EXACTLY the static tau at init (exp(tanh(0)) == 1); no hidden init difference")

    with torch.no_grad():
        liquid_only.tau_module.gate_v.fill_(0.7)
        liquid_only.tau_module.gate_b.fill_(-0.3)
    t_moved = liquid_only.tau_module.tau(liquid_only.neuron_type, v=v, u=u)
    assert not torch.allclose(t_moved, t_static)
    base_rep = liquid_only.tau_module.base_tau()[liquid_only.neuron_type]
    assert bool(((t_moved >= base_rep / math.e - 1e-6) & (t_moved <= base_rep * math.e + 1e-6)).all())
    ok("after gate movement tau is input-dependent and stays bounded in [tau/e, tau*e]")
    with torch.no_grad():
        liquid_only.tau_module.gate_v.zero_()
        liquid_only.tau_module.gate_b.zero_()

    # -- D7: parameter accounting -------------------------------------------
    print("\n-- D7: parameter accounting --")
    rb = baseline.parameter_report()
    rl = both.parameter_report()
    assert rb["tau"] == con.n_types, rb
    assert rl["tau"] == 4 * con.n_types, rl
    assert rl["total"] - rb["total"] == 3 * con.n_types
    ok(f"synthetic: static tau {rb['tau']} params, liquid tau {rl['tau']} params (raw + 3 gates)")

    # The real flyvis shape, from verified constants rather than by hand.
    n_pairs = FLYVIS.n_type_pairs_checkpoint
    n_types = FLYVIS.n_types
    real_baseline = n_pairs + n_types + n_types
    real_liquid = n_pairs + 4 * n_types + n_types
    assert real_baseline == 734, real_baseline
    assert real_liquid == 929, real_liquid
    assert real_liquid - real_baseline == 195
    pct = 100.0 * (real_liquid - real_baseline) / real_baseline
    assert abs(pct - 26.567) < 0.01, pct
    ok(f"real flyvis shape: baseline {real_baseline} -> constrained {real_liquid} (+{pct:.1f}%), filler {real_liquid - real_baseline}")
    ok("matches D7 exactly (929, +26.6%, filler 195) -- NOT the round-1 figure of 767/+4%")

    assert both.parameter_matched_filler() == 3 * con.n_types
    ok(f"parameter_matched_filler() = {both.parameter_matched_filler()} on the synthetic shape")

    r_edge = build_arm(con, dale=True, liquid_tau=True, tying="per_edge").parameter_report()
    r_post = build_arm(con, dale=True, liquid_tau=True, tying="per_post_type").parameter_report()
    assert r_edge["magnitude"] == con.n_edges
    assert r_post["magnitude"] == con.n_types
    assert r_edge["reduction_factor"] == 1.0
    ok(f"tying granularity works: per_edge {r_edge['magnitude']} / per_type_pair {rl['magnitude']} / per_post_type {r_post['magnitude']} magnitudes")

    # -- dynamics -------------------------------------------------------------
    print("\n-- dynamics --")
    v0 = torch.zeros(con.n_neurons)
    x = torch.zeros(con.n_neurons)
    x[0] = 1.0
    for m, name in ((baseline, "baseline"), (both, "both")):
        v = v0.clone()
        for _ in range(50):
            v = m(v, x)
        assert v.shape == (con.n_neurons,)
        assert bool(torch.isfinite(v).all()), f"{name} produced non-finite state"
    ok("50-step rollout finite and correctly shaped in both arms")

    v = v0.clone()
    for _ in range(5):
        v = both(v, x)
    loss = v.pow(2).sum()
    loss.backward()
    grads = {n: p.grad for n, p in both.named_parameters() if p.grad is not None}
    assert "weights.raw_magnitude" in grads
    assert any("gate" in n for n in grads), sorted(grads)
    assert all(bool(torch.isfinite(g).all()) for g in grads.values())
    ok(f"gradients flow to {len(grads)} parameter tensors, all finite (incl. the CfC gates)")

    for m in (baseline, both):
        assert not m.weights.signs.requires_grad
        assert not m.weights.syn_count.requires_grad
    ok("signs and synapse counts are fixed buffers, not parameters")

    # -- ParameterReport is the single source of the arithmetic (D7) --------
    print("\n-- ParameterReport / integrator table / tau-vs-dt --")
    pr_real = ParameterReport.for_fly_connectome(liquid_tau=True)
    assert (pr_real.total, pr_real.baseline_equivalent, pr_real.delta) == (929, 734, 195)
    assert abs(pr_real.pct - 26.567) < 0.01
    ok(f"ParameterReport.for_fly_connectome(): {pr_real.describe()}")
    assert ParameterReport.for_fly_connectome(liquid_tau=False).total == 734
    ok("static-tau variant reproduces flyvis's 734 exactly")

    for m, name in ((baseline, "baseline"), (both, "both")):
        ParameterReport.from_module(m).check_against_module(m)
    ok("check_against_module(): report == sum(p.numel()) for both arms; cannot drift")

    # -- integrator table matches the docstring -----------------------------
    rows = integrator_comparison()
    row = next(r for r in rows if abs(r.tau - FLYVIS.tau_init) < 1e-12)
    assert abs(row.euler - 0.600000) < 1e-6 and abs(row.exact - 0.670320) < 1e-6
    assert abs(row.rel_diff - 0.1172) < 1e-3
    ok(f"integrator_comparison() at tau={row.tau}: euler {row.euler:.6f} vs exact {row.exact:.6f} ({row.rel_diff:.2%})")
    dead = [r for r in rows if r.euler_gradient_is_dead]
    assert all(r.tau < FLYVIS.dt for r in dead) and len(dead) >= 2
    ok(f"{len(dead)}/{len(rows)} table rows sit below dt, where flyvis's Euler guard kills d/dtau")

    tdr = tau_dt_report(both)
    assert tdr.n_types == con.n_types and tdr.n_below_dt == 0
    ok(f"tau_dt_report(): {tdr.describe()}")

    # -- pre-registered arms and the honesty manifest -----------------------
    assert PREREGISTERED_ARMS["baseline"] == BASELINE_ARM
    assert len([k for k, v in CONSTRAINED_ARM.axes.items() if BASELINE_ARM.axes[k] != v]) == 2
    ok("PREREGISTERED_ARMS: baseline vs constrained differ in 2 axes (documented, not hidden)")
    assert set(ARM_AXES) == set(BASELINE_ARM.axes)
    ok(f"ARM_AXES matches ArmSpec.axes: {ARM_AXES}")

    import dale_typed_cfc as _self
    missing = [n for n in _self.__all__ if not hasattr(_self, n)]
    assert not missing, f"__all__ promises absent names: {missing}"
    ok(f"__all__ is honest: all {len(_self.__all__)} exported names exist")
    assert all(not hasattr(_self, n) for n in NOT_IMPLEMENTED)
    ok(f"NOT_IMPLEMENTED lists {len(NOT_IMPLEMENTED)} names absent from this module (seam: flyvis_seam.py)")

    print("\n" + line)
    print(f"ALL {checks} CHECKS PASSED in {_time.perf_counter() - t0:.2f} s")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
