"""The flyvis seam: inject this project's parameterization into `flyvis.Network`.

`dale_typed_cfc.py` implements the parameterization standalone, against a
:class:`~dale_typed_cfc.TypedConnectome`. Stage 2 has to train INSIDE flyvis so that the
connectome, task, dataset, decoder, optimizer and schedule are flyvis's own. This module
is that adapter. Everything that reaches `flyvis.Network` goes through
:func:`flyvis_arm_config` and the adapter classes below; :func:`verify_seam` is the
acceptance test and must pass, in the runtime's own device regime, before any metered run.

===============================================================================
HOW FLYVIS DISPATCH WORKS  (flyvis v1.2.0 = 92b3845)
===============================================================================

flyvis instantiates configurable components by CLASS NAME::

    def find_subclass(cls, target_subclass_name):
        for subclass in cls.__subclasses__():          # depth-first
            if subclass.__qualname__ == target_subclass_name:
                return subclass
            ...

    def forward_subclass(cls, config={}, subclass_key="type", unpack_kwargs=True):
        target_subclass = config.pop(subclass_key, None)
        kwargs = {k: v for k, v in config.items() if k != subclass_key}
        subclass = find_subclass(cls, target_subclass)  # falls back to cls with a warning
        instance = object.__new__(subclass)
        instance.__init__(**kwargs)

Consequences that shape this module:

1. The match is on ``__qualname__``. A class defined inside a function has qualname
   ``f.<locals>.Name`` and is never found; flyvis then silently builds the BASE class.
   All adapters are therefore defined at MODULE level (audit K1), and the module keeps
   strong references to them (K2).
2. ``__subclasses__()`` holds weak references, and the FIRST match wins. After
   ``importlib.reload`` the old classes stay findable until garbage-collected -- forever,
   if an instance is alive. :func:`register_flyvis_adapters` refuses to report success
   unless every adapter name resolves to exactly one class, and that class is this
   module's.
3. Every remaining config key is passed as a kwarg. `Value(**kwargs)` swallows unknown
   keys, so a config flag that nothing reads is a silent no-op (audit I1: the old
   ``_dale_override`` flag). Treatments here are therefore always expressed as a CLASS
   (``type:``) that does the work in ``__init__``, never as a flag.

===============================================================================
WHAT EACH ARM IS, IN FLYVIS TERMS
===============================================================================

Shared substrate (identical in every arm -- D1):

* ``edge_config.syn_strength.initial_dist = SoftplusValue`` with ``clamp: None``.
  flyvis's ``clamp: non_negative`` manufactures the exact-zero point mass (418/604
  magnitudes exactly 0 in some released seeds). :class:`SoftplusValue` stores the
  inverse-softplus of flyvis's own initial value, so the semantic initialization is
  unchanged (audit I4), and reads through a ``clone()`` so flyvis's in-place optimizer
  step before its activity-penalty backward does not invalidate the graph (I6).
  ``clamp`` must be ``None``: `Network.clamp` raises on ``False`` (I5).
* ``dynamics`` is :class:`ExactExponentialDynamics` or its subclass
  :class:`CfCDynamics` -- see the integrator section.
* ``node_config.time_const.clamp = [TAU_MIN, TAU_MAX]``: flyvis's own projected-clamp
  mode. Stock flyvis has no clamp; it guards in the forward with ``max(tau, dt)``, which
  zeroes d/dtau below dt. The exact integrator needs no such guard but must keep tau
  positive; the floor sits where d(rate)/d(tau) is still O(10) at dt=0.02 (I17).
* Parameter matching: an arm without CfC gates carries three per-type
  :class:`InertFiller` node parameters of the same shape (3 x n_types), requires_grad
  True and never read by the dynamics, so both arms report the same free-parameter count
  (929 at extent 15). Their gradients are ``None`` and Adam skips them.

Treatments (the ONLY config paths allowed to differ -- :data:`TREATMENT_PREFIXES`):

* Dale: ``edge_config.sign.type = DaleSynapseSign`` plus ``dale_signs`` (published
  neurotransmitter signs, written INTO the config so a checkpoint is self-describing)
  and ``dale_scope``. :class:`DaleSynapseSign` runs flyvis's own `SynapseSign`
  construction, then rewrites the 604 group values so every source type has one sign.
* Liquid tau: ``dynamics.type = CfCDynamics`` plus ``node_config.gate_{v,u,b}``, per-type
  :class:`CfCGate` parameters (65 each). They are ordinary flyvis node parameters:
  created at construction (so the optimizer, built before the first forward, sees them
  -- I2), grouped by type (not per neuron -- I3), and saved/loaded with the state dict
  (I8).

===============================================================================
THE INTEGRATOR  (D5)
===============================================================================

flyvis asks a dynamics class for a VELOCITY and integrates with explicit Euler::

    self.dynamics.write_state_velocity(vel, state, params, self.target_sum, x_t, dt=dt)
    next = state.nodes[k] + vel.nodes[k] * dt

`PPNeuronIGRSynapses` writes ``vel = 1/max(tau, dt) * (-v + u)`` with
``u = bias + sum(weight * relu(v_source)) + x_t``, i.e. retention ``1 - dt/max(tau,dt)``.

The exact (zero-order-hold) solution is reachable by changing ONLY the coefficient::

    v_next = e^(-dt/tau) v + (1 - e^(-dt/tau)) u = v + dt * [(1 - e^(-dt/tau)) / dt] * (u - v)

At dt=0.02, tau=0.05: Euler retention 0.600000, exact 0.670320 (11.72% apart). As
dt/tau -> 0 the coefficient -> 1/tau. It is computed as ``(1 - exp(-dt/tau)) / dt``
rather than with ``expm1``: in float32 the backward of expm1 underflows to exactly 0 for
tau below ~dt/17, whereas exp's backward stays non-zero down to ~dt/100 (I17). The
float limit near tau ~ dt/100 remains; :data:`TAU_MIN` keeps tau above it.

CfC: ``tau_eff = tau * exp(tanh(gate_v*v + gate_u*u + gate_b))``, per cell type, bounded
to ``[tau/e, tau*e]``. Gates initialize to exactly 0, so ``exp(tanh(0)) == 1`` and arm B's
initial dynamics are bit-identical to the Dale-only arm's; :func:`verify_seam` checks it.

===============================================================================
VERIFICATION
===============================================================================

:func:`verify_seam` builds real networks and checks, by identity and on tensors: adapter
dispatch, clamp configuration, free-parameter counts, that the optimizer built before the
first forward holds every trainable tensor, softplus init equal to flyvis's, Dale sign
flips (count, and one sign per source type), the exact-integrator update against its
closed form, zero-gate bit-identity, finite bounded activity, non-zero gate gradients,
inert filler, flyvis's step-then-penalty-backward order, and a strict state-dict round
trip. It reports ``ok=False`` whenever flyvis is unavailable -- there is no config-only
pass.
"""

from __future__ import annotations

import copy
import gc
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SOFTPLUS_VALUE_NAME",
    "EXACT_DYNAMICS_NAME",
    "CFC_DYNAMICS_NAME",
    "DALE_SIGN_NAME",
    "CFC_GATE_NAME",
    "FILLER_NAME",
    "GATE_KEYS",
    "FILLER_KEYS",
    "TAU_MIN",
    "TAU_MAX",
    "ARM_PRESETS",
    "exact_rate",
    "euler_rate",
    "retention",
    "rate_comparison",
    "inverse_softplus",
    "DaleSignPlan",
    "flyvis_arm_config",
    "flyvis_network_kwargs",
    "apply_arm_to_solver_config",
    "deep_merge",
    "config_diff",
    "assert_configs_share_substrate",
    "SUBSTRATE_REQUIREMENTS",
    "TREATMENT_PREFIXES",
    "register_flyvis_adapters",
    "adapter_conflicts",
    "trainable_parameter_counts",
    "semantic_from_state_dict",
    "verify_seam",
    "SEAM_STATUS",
]

#: Class names exactly as flyvis looks them up (``__qualname__``).
SOFTPLUS_VALUE_NAME = "SoftplusValue"
EXACT_DYNAMICS_NAME = "ExactExponentialDynamics"
CFC_DYNAMICS_NAME = "CfCDynamics"
DALE_SIGN_NAME = "DaleSynapseSign"
CFC_GATE_NAME = "CfCGate"
FILLER_NAME = "InertFiller"

GATE_KEYS: Tuple[str, ...] = ("gate_v", "gate_u", "gate_b")
FILLER_KEYS: Tuple[str, ...] = ("filler_v", "filler_u", "filler_b")

#: Projected clamp on the per-type time constant (seconds), shared by every arm.
TAU_MIN = 2e-3
TAU_MAX = 1e3

#: Forward-pass floor inside the rate. Never binds while the projected clamp is active;
#: it only keeps a network loaded without that clamp from producing inf/NaN.
_TAU_EPS = 1e-6

DALE_SCOPES = ("all", "unambiguous_only")

#: Named arms. "C" is kept for the pre-registration record; on the fib25/fib19 connectome
#: its signs equal arm A's (every unambiguous type already emits one sign), which
#: :class:`DaleSignPlan` reports as ``n_flipped == 0``.
ARM_PRESETS: Dict[str, Dict[str, Any]] = {
    "A": dict(dale=False, liquid_tau=False),
    "B": dict(dale=True, liquid_tau=True),
    "dale_only": dict(dale=True, liquid_tau=False),
    "liquid_only": dict(dale=False, liquid_tau=True),
    "C": dict(dale=True, dale_scope="unambiguous_only", liquid_tau=True),
}


# =============================================================================
# the integrator arithmetic -- pure, testable without flyvis
# =============================================================================


def exact_rate(dt: float, tau: float) -> float:
    """Velocity coefficient giving the exact zero-order-hold update: ``(1-exp(-dt/tau))/dt``."""
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    return -math.expm1(-dt / tau) / dt


def euler_rate(dt: float, tau: float) -> float:
    """flyvis's coefficient: ``1 / max(tau, dt)``. Reference only."""
    return 1.0 / max(tau, dt)


def retention(dt: float, rate: float) -> float:
    """Per-step retention implied by a velocity coefficient: ``1 - dt*rate``."""
    return 1.0 - dt * rate


def rate_comparison(dt: float = 0.02, taus: Optional[Sequence[float]] = None) -> List[Dict[str, float]]:
    """Table of Euler vs exact retention, and where Euler's guard kills d/dtau."""
    if taus is None:
        taus = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5]
    rows = []
    for tau in taus:
        ke, kx = retention(dt, euler_rate(dt, tau)), retention(dt, exact_rate(dt, tau))
        rows.append({
            "tau": float(tau), "dt": float(dt),
            "euler_retention": ke, "exact_retention": kx,
            "rel_diff": abs(kx - ke) / max(abs(ke), 1e-12),
            "euler_gradient_dead": float(tau) < float(dt),
        })
    return rows


def inverse_softplus(y: float) -> float:
    """``x`` with ``softplus(x) == y``, stable for all ``y > 0``: ``y + log(-expm1(-y))``."""
    if not y > 0.0:
        raise ValueError(f"inverse_softplus needs y > 0, got {y}")
    return y + math.log(-math.expm1(-y))


# =============================================================================
# Dale's law at the level flyvis stores signs: 604 (source_type, target_type) groups
# =============================================================================


@dataclass
class DaleSignPlan:
    """Resolved, auditable sign assignment for flyvis's per-(source, target) sign groups.

    Pure Python and group-level (604 entries, not 1.5M edges -- audit K4). Used by
    :class:`DaleSynapseSign` at construction and attached to it as ``.dale_plan``, so the
    exact flips a network was built with can be read back and committed.

    Resolution, per source type ``t`` with the set ``S_t`` of signs its groups carry:

    * ``t`` in ``explicit_signs`` and ``len(S_t) == 1`` and they disagree -> ``ValueError``
      (a published transmitter contradicting a consistent connectome is a data conflict,
      not something to resolve silently).
    * scope ``"all"``: explicit sign if given; else the single sign in ``S_t``; a
      mixed-sign type with no explicit sign -> ``ValueError``.
    * scope ``"unambiguous_only"``: mixed-sign types keep their per-pair signs; all
      other types keep their (single) sign. On this connectome that changes nothing.
    """

    keys: List[Tuple[str, str]]
    before: List[float]
    after: List[float]
    scope: str
    type_signs: Dict[str, float]
    mixed_types: Tuple[str, ...]
    explicit_signs: Dict[str, float]
    group_weight: Optional[List[float]] = None

    @classmethod
    def from_groups(
        cls,
        keys: Sequence[Tuple[str, str]],
        signs: Sequence[float],
        explicit_signs: Optional[Mapping[str, float]] = None,
        *,
        scope: str = "all",
        group_weight: Optional[Sequence[float]] = None,
    ) -> "DaleSignPlan":
        if scope not in DALE_SCOPES:
            raise ValueError(f"dale_scope must be one of {DALE_SCOPES}, got {scope!r}")
        keys = [(str(s), str(t)) for s, t in keys]
        before = [float(x) for x in signs]
        if len(keys) != len(before):
            raise ValueError(f"{len(keys)} keys but {len(before)} signs")
        if any(x not in (-1.0, 1.0) for x in before):
            raise ValueError("connectome signs must all be +1 or -1")
        explicit = {str(k): float(v) for k, v in dict(explicit_signs or {}).items()}
        if any(v not in (-1.0, 1.0) for v in explicit.values()):
            raise ValueError(f"explicit signs must be +1 or -1: {explicit}")
        if group_weight is not None and len(group_weight) != len(keys):
            raise ValueError("group_weight must have one entry per group")

        by_type: Dict[str, set] = {}
        for (src, _), s in zip(keys, before):
            by_type.setdefault(src, set()).add(s)
        mixed = tuple(sorted(t for t, ss in by_type.items() if len(ss) > 1))

        type_signs: Dict[str, float] = {}
        for t, ss in by_type.items():
            if len(ss) == 1:
                (only,) = ss
                if t in explicit and explicit[t] != only:
                    raise ValueError(
                        f"published sign for {t} is {explicit[t]:+.0f} but every {t} group in "
                        f"the connectome is {only:+.0f}; refusing to resolve this silently")
                type_signs[t] = only
            elif t in explicit:
                type_signs[t] = explicit[t]
            elif scope == "all":
                raise ValueError(
                    f"source type {t} emits both signs and has no published sign; Dale's law "
                    f"is undefined for it. Add it to dale_signs or use dale_scope="
                    f"'unambiguous_only'.")

        if scope == "all":
            after = [type_signs[src] for src, _ in keys]
        else:
            after = [s if src in mixed else type_signs[src] for (src, _), s in zip(keys, before)]

        return cls(
            keys=keys, before=before, after=after, scope=scope,
            type_signs=type_signs, mixed_types=mixed, explicit_signs=explicit,
            group_weight=None if group_weight is None else [float(w) for w in group_weight],
        )

    @property
    def flipped(self) -> List[Tuple[str, str]]:
        return [k for k, b, a in zip(self.keys, self.before, self.after) if a != b]

    @property
    def n_flipped(self) -> int:
        return len(self.flipped)

    @property
    def pct_weight_flipped(self) -> Optional[float]:
        if self.group_weight is None:
            return None
        total = sum(self.group_weight)
        flip = sum(w for w, b, a in zip(self.group_weight, self.before, self.after) if a != b)
        return 100.0 * flip / total if total > 0 else 0.0

    def mixed_after(self) -> List[str]:
        """Source types that still emit both signs after the plan (empty under scope 'all')."""
        seen: Dict[str, set] = {}
        for (src, _), s in zip(self.keys, self.after):
            seen.setdefault(src, set()).add(s)
        return sorted(t for t, ss in seen.items() if len(ss) > 1)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "n_groups": len(self.keys),
            "n_flipped": self.n_flipped,
            "flipped": [f"{a}->{b}" for a, b in self.flipped],
            "pct_weight_flipped": self.pct_weight_flipped,
            "mixed_types_before": list(self.mixed_types),
            "mixed_types_after": self.mixed_after(),
            "explicit_signs": dict(self.explicit_signs),
        }

    def describe(self) -> str:
        d = self.as_dict()
        pct = "n/a" if d["pct_weight_flipped"] is None else f"{d['pct_weight_flipped']:.2f}%"
        return "\n".join([
            f"Dale sign plan (scope={self.scope})",
            f"  groups                : {d['n_groups']}",
            f"  groups flipped        : {d['n_flipped']}",
            f"  synapse weight flipped: {pct}",
            f"  mixed types before    : {d['mixed_types_before']}",
            f"  mixed types after     : {d['mixed_types_after']}",
            "  flips: " + (", ".join(d["flipped"]) or "(none)"),
        ])


# =============================================================================
# arm configs
# =============================================================================


def _published_signs() -> Dict[str, float]:
    import dale_typed_cfc as dtc  # local: keeps this module importable without torch

    return {k: float(v) for k, v in dtc.published_explicit_signs().items()}


def _gate_entry(type_name: str) -> Dict[str, Any]:
    return {
        "type": type_name,
        "groupby": ["type"],
        "initial_dist": "Value",
        "value": 0.0,
        "requires_grad": True,
        "clamp": None,
    }


def flyvis_arm_config(
    *,
    dale: bool,
    liquid_tau: bool,
    dale_scope: str = "all",
    dale_signs: Optional[Mapping[str, float]] = None,
    param_matched: bool = True,
    tying: Sequence[str] = ("source_type", "target_type"),
    tau_init: float = 0.05,
    tau_clamp: Tuple[float, float] = (TAU_MIN, TAU_MAX),
    magnitude_scale: float = 0.01,
    activation: Any = "relu",
) -> Dict[str, Any]:
    """The flyvis config OVERRIDES for one arm (``dynamics``, ``node_config``, ``edge_config``).

    Partial by design: merge onto flyvis's defaults with :func:`flyvis_network_kwargs`
    (for ``Network(**kwargs)``) or :func:`apply_arm_to_solver_config` (for
    ``MultiTaskSolver``). Passing these sections to flyvis directly REPLACES its defaults
    (e.g. drops ``syn_count``) -- audit I19.

    There is no keyword that disables the softplus magnitudes, the exact integrator or
    the tau clamp: those are the shared substrate (D1).
    """
    tying = list(tying)
    if not {"source_type", "target_type"} <= set(tying):
        raise ValueError(
            "SynapseCountScaling reads grouped source_type/target_type columns, so tying "
            f"must include both; got {tying}")
    if dale_scope not in DALE_SCOPES:
        raise ValueError(f"dale_scope must be one of {DALE_SCOPES}, got {dale_scope!r}")
    lo, hi = (float(x) for x in tau_clamp)
    if not 0.0 < lo <= float(tau_init) <= hi:
        raise ValueError(f"need 0 < tau_clamp[0] <= tau_init <= tau_clamp[1]; got {tau_clamp}, {tau_init}")

    act = {"type": activation} if isinstance(activation, str) else copy.deepcopy(dict(activation))

    node_config: Dict[str, Any] = {
        "time_const": {
            "type": "TimeConstant",
            "groupby": ["type"],
            "initial_dist": "Value",
            "value": float(tau_init),
            "requires_grad": True,
            "clamp": [lo, hi],                       # SHARED SUBSTRATE
        },
    }
    if liquid_tau:
        node_config.update({k: _gate_entry(CFC_GATE_NAME) for k in GATE_KEYS})
    elif param_matched:
        node_config.update({k: _gate_entry(FILLER_NAME) for k in FILLER_KEYS})

    sign: Dict[str, Any] = {
        "type": DALE_SIGN_NAME if dale else "SynapseSign",
        "form": "value",
        "requires_grad": False,
        "initial_dist": "Value",
        "groupby": ["source_type", "target_type"],
    }
    if dale:
        sign["dale_scope"] = dale_scope
        sign["dale_signs"] = {str(k): float(v) for k, v in dict(dale_signs or _published_signs()).items()}

    return {
        "dynamics": {
            "type": CFC_DYNAMICS_NAME if liquid_tau else EXACT_DYNAMICS_NAME,
            "activation": act,
        },
        "node_config": node_config,
        "edge_config": {
            "syn_strength": {
                "type": "SynapseCountScaling",
                "initial_dist": SOFTPLUS_VALUE_NAME,   # SHARED SUBSTRATE
                "requires_grad": True,
                "scale": float(magnitude_scale),
                "clamp": None,                          # SHARED SUBSTRATE; never False (I5)
                "groupby": tying,
            },
            "sign": sign,
        },
    }


def _plain(x: Any) -> Any:
    """Recursively convert Namespaces/dicts to plain dicts (lists and scalars deep-copied)."""
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return copy.deepcopy(x)


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    """``base`` with ``over`` merged in recursively; neither input is modified.

    The ``type`` of a component and the whole ``dynamics`` section are REPLACED rather
    than merged, so no key of the replaced class (e.g. a stale ``dale_signs``) can leak
    into the new one.
    """
    out = _plain(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict) and k != "dynamics":
            if "type" in v and out[k].get("type") != v["type"]:
                out[k] = _plain(v)
            else:
                out[k] = deep_merge(out[k], v)
        else:
            out[k] = _plain(v)
    return out


_NETWORK_SECTIONS = ("connectome", "dynamics", "node_config", "edge_config", "stimulus_config")


def flyvis_network_kwargs(extent: Optional[int] = None, **arm_kw: Any) -> Dict[str, Any]:
    """Complete ``Network(**kwargs)`` for one arm: flyvis's own `Network.__init__` defaults
    deep-merged with :func:`flyvis_arm_config`. Requires flyvis."""
    import inspect

    from flyvis.network.network import Network

    sig = inspect.signature(Network.__init__).parameters
    over = flyvis_arm_config(**arm_kw)
    out = {k: deep_merge(sig[k].default, over.get(k, {})) for k in _NETWORK_SECTIONS}
    out["dynamics"] = _plain(over["dynamics"])          # replaced whole, never merged
    if extent is not None:
        out["connectome"]["extent"] = int(extent)
    for need in ("sign", "syn_count", "syn_strength"):
        if need not in out["edge_config"]:
            raise AssertionError(f"merge lost edge_config.{need}")
    for need in ("bias", "time_const"):
        if need not in out["node_config"]:
            raise AssertionError(f"merge lost node_config.{need}")
    return out


def apply_arm_to_solver_config(config: Any, extent: Optional[int] = None, **arm_kw: Any) -> Any:
    """Return a COPY of a flyvis solver config whose ``network`` section is this arm.

    ``config`` is what ``flyvis.utils.config_utils.get_default_config`` returns. The
    result is a datamate ``Namespace`` ready for ``MultiTaskSolver(name, config)``.
    """
    from datamate import namespacify

    cfg = _plain(config)
    over = flyvis_arm_config(**arm_kw)
    net = cfg["network"]
    net["dynamics"] = _plain(over["dynamics"])            # replaced whole, never merged
    for k in ("node_config", "edge_config"):
        net[k] = deep_merge(net.get(k, {}), over[k])
    if extent is not None:
        net["connectome"]["extent"] = int(extent)
    return namespacify(cfg)


def _dig(cfg: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur:
            return KeyError
        cur = cur[k]
    return cur


def config_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[Tuple[str, Any, Any]]:
    """Every (dotted path, a_value, b_value) where two configs differ. Missing -> ``KeyError``."""
    out: List[Tuple[str, Any, Any]] = []

    def walk(x: Any, y: Any, path: Tuple[str, ...]) -> None:
        if isinstance(x, Mapping) and isinstance(y, Mapping):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k, KeyError), y.get(k, KeyError), path + (str(k),))
        elif _plain(x) != _plain(y):
            out.append((".".join(path), x, y))

    walk(a, b, ())
    return out


#: Paths that may differ between arms -- the treatments, and nothing else. Prefix match.
TREATMENT_PREFIXES: Tuple[str, ...] = (
    "dynamics.type",
    "edge_config.sign.type",
    "edge_config.sign.dale_signs",
    "edge_config.sign.dale_scope",
    "edge_config.syn_strength.groupby",
    *(f"node_config.{k}" for k in GATE_KEYS),
    *(f"node_config.{k}" for k in FILLER_KEYS),
)

#: (path, predicate, description) every arm must satisfy -- the substrate is not just
#: "equal between arms" but "equal AND ours" (two stock configs would otherwise pass).
SUBSTRATE_REQUIREMENTS: Tuple[Tuple[Tuple[str, ...], Any, str], ...] = (
    (("edge_config", "syn_strength", "initial_dist"), lambda v: v == SOFTPLUS_VALUE_NAME, f"== {SOFTPLUS_VALUE_NAME}"),
    (("edge_config", "syn_strength", "clamp"), lambda v: v is None, "is None"),
    (("dynamics", "type"), lambda v: v in (EXACT_DYNAMICS_NAME, CFC_DYNAMICS_NAME), "is an exact-integrator class"),
    (("node_config", "time_const", "clamp"),
     lambda v: isinstance(v, (list, tuple)) and len(v) == 2 and 0 < float(v[0]) < float(v[1]), "is [lo>0, hi]"),
)


def _is_treatment(path: str) -> bool:
    return any(path == p or path.startswith(p + ".") for p in TREATMENT_PREFIXES)


def _arm_consistency(cfg: Mapping[str, Any], label: str) -> Dict[str, Any]:
    dyn = _dig(cfg, ("dynamics", "type"))
    nodes = _dig(cfg, ("node_config",))
    nodes = nodes if isinstance(nodes, Mapping) else {}
    gates = [k for k in GATE_KEYS if k in nodes]
    fillers = [k for k in FILLER_KEYS if k in nodes]
    liquid = dyn == CFC_DYNAMICS_NAME
    if liquid and len(gates) != len(GATE_KEYS):
        raise AssertionError(f"{label}: dynamics is {CFC_DYNAMICS_NAME} but gates present = {gates}")
    if not liquid and gates:
        raise AssertionError(f"{label}: gates {gates} present but dynamics is {dyn}")
    if gates and fillers:
        raise AssertionError(f"{label}: both gates and filler present")
    for k in gates + fillers:
        want = CFC_GATE_NAME if k in GATE_KEYS else FILLER_NAME
        if nodes[k].get("type") != want or nodes[k].get("groupby") != ["type"]:
            raise AssertionError(f"{label}: node_config.{k} must be a per-type {want}")
    sign_t = _dig(cfg, ("edge_config", "sign", "type"))
    has_signs = _dig(cfg, ("edge_config", "sign", "dale_signs")) is not KeyError
    dale = sign_t == DALE_SIGN_NAME
    if dale != has_signs:
        raise AssertionError(f"{label}: sign type {sign_t} but dale_signs present = {has_signs}")
    return {"liquid": liquid, "dale": dale, "n_extra_node_params": len(gates) + len(fillers)}


def assert_configs_share_substrate(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    require_param_matched: bool = True,
) -> Dict[str, Any]:
    """Raise unless both configs carry OUR substrate and differ only in treatment paths.

    Works on partial arm configs and on full merged configs. Checks, in order:
    every :data:`SUBSTRATE_REQUIREMENTS` in each arm; internal consistency of each arm's
    declared treatments (CfC <-> gates, DaleSynapseSign <-> dale_signs); equal count of
    extra per-type node parameters (unless ``require_param_matched=False``); and no
    difference outside :data:`TREATMENT_PREFIXES`.
    """
    for label, cfg in (("a", a), ("b", b)):
        for path, pred, desc in SUBSTRATE_REQUIREMENTS:
            v = _dig(cfg, path)
            if v is KeyError or not pred(v):
                raise AssertionError(f"shared substrate violated in arm {label}: {'.'.join(path)} = {v!r}, required {desc}")
    ca, cb = _arm_consistency(a, "a"), _arm_consistency(b, "b")
    if require_param_matched and ca["n_extra_node_params"] != cb["n_extra_node_params"]:
        raise AssertionError(
            f"parameter matching violated: arm a has {ca['n_extra_node_params']} extra per-type "
            f"node parameters, arm b has {cb['n_extra_node_params']}")
    diffs = config_diff(a, b)
    unexpected = [d for d in diffs if not _is_treatment(d[0])]
    if unexpected:
        raise AssertionError(
            "arm configs differ outside the declared treatments: "
            + "; ".join(f"{p}: {x!r} != {y!r}" for p, x, y in unexpected[:8]))
    axes = []
    if ca["dale"] != cb["dale"] or _dig(a, ("edge_config", "sign", "dale_scope")) != _dig(b, ("edge_config", "sign", "dale_scope")):
        axes.append("dale")
    if ca["liquid"] != cb["liquid"]:
        axes.append("liquid_tau")
    if _dig(a, ("edge_config", "syn_strength", "groupby")) != _dig(b, ("edge_config", "syn_strength", "groupby")):
        axes.append("tying")
    return {"differing_paths": [d[0] for d in diffs], "axes": axes}


# =============================================================================
# the adapters -- MODULE LEVEL, so __qualname__ is the bare name (K1) and the module
# holds strong references (K2)
# =============================================================================

try:
    import torch
    from torch import nn
    from flyvis.network.dynamics import NetworkDynamics, PPNeuronIGRSynapses
    from flyvis.network.initialization import (
        InitialDistribution,
        Parameter,
        SynapseSign,
        TimeConstant,
        Value,
    )

    _FLYVIS_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:  # pragma: no cover - depends on the environment
    _FLYVIS_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


if _FLYVIS_IMPORT_ERROR is None:

    def _inverse_softplus_tensor(y: "torch.Tensor") -> "torch.Tensor":
        return y + torch.log(-torch.expm1(-y))

    class SoftplusValue(Value):
        """Magnitude reparametrization ``semantic = softplus(raw)``. SHARED SUBSTRATE.

        Stores ``inverse_softplus`` of the value flyvis would have used, so the SEMANTIC
        initialization equals stock flyvis's (audit I4). Reads through ``clone()`` because
        softplus saves its input for backward, and flyvis steps the optimizer in place
        before its activity-penalty backward reuses the graph (I6).
        """

        def __init__(self, value: Any, requires_grad: bool, clamp: Any = None, **kwargs: Any) -> None:
            if clamp is not None:
                raise ValueError(
                    f"SoftplusValue requires clamp: None, got {clamp!r}. A clamp acts on RAW "
                    "values (unbounded below under softplus), and flyvis's Network.clamp raises "
                    "on False.")
            Value.__init__(self, value, requires_grad, clamp=None, **kwargs)
            with torch.no_grad():
                y = self.raw_values
                if not bool(torch.isfinite(y).all()) or bool((y <= 0).any()):
                    raise ValueError("SoftplusValue needs finite, strictly positive initial values")
                self.raw_values.copy_(_inverse_softplus_tensor(y))

        @property
        def semantic_values(self) -> "torch.Tensor":
            return nn.functional.softplus(self.raw_values.clone())

    class CfCGate(TimeConstant):
        """Per-cell-type CfC gate (arm B). flyvis's own `TimeConstant` construction, config
        ``value: 0.0``: one entry per type, created at network construction."""

    class InertFiller(TimeConstant):
        """Per-cell-type parameter that no dynamics reads. Makes arms without CfC gates
        parameter-matched. Its gradient is always ``None``; :func:`verify_seam` checks it."""

    class DaleSynapseSign(SynapseSign):
        """flyvis's `SynapseSign`, then Dale's law over its (source, target) groups.

        Config keys (beyond `SynapseSign`'s): ``dale_signs`` ``{type: +-1}`` and
        ``dale_scope`` (``"all"`` | ``"unambiguous_only"``). The resolved
        :class:`DaleSignPlan` is kept as ``self.dale_plan``.
        """

        def __init__(self, param_config: Any, connectome: Any) -> None:
            import numpy as np

            SynapseSign.__init__(self, param_config, connectome)  # deep-copies its config
            raw = self.parameter.raw_values
            idx = self.indices.detach().cpu().numpy()
            n_syn = np.asarray(connectome.edges.n_syn[:], dtype=np.float64)
            weight = np.bincount(idx, weights=n_syn, minlength=len(self.keys))
            plan = DaleSignPlan.from_groups(
                self.keys,
                raw.detach().cpu().tolist(),
                param_config.get("dale_signs", None),
                scope=param_config.get("dale_scope", "all"),
                group_weight=weight.tolist(),
            )
            with torch.no_grad():
                raw.copy_(torch.tensor(plan.after, dtype=raw.dtype, device=raw.device))
            self.dale_plan = plan

    class ExactExponentialDynamics(PPNeuronIGRSynapses):
        """flyvis's passive point neurons with the EXACT zero-order-hold update. All arms.

        Inherits `PPNeuronIGRSynapses`'s derived weights, initial state and currents
        unchanged; only the velocity coefficient differs. Activation is built exactly as
        flyvis does (`activation_fns[type](**kwargs)`, audit I22). Unknown config keys
        raise instead of being ignored.
        """

        def __init__(self, activation: Optional[Mapping[str, Any]] = None) -> None:
            act = dict(activation) if activation is not None else {"type": "relu"}
            NetworkDynamics.__init__(self, activation=act)

        def tau_effective(self, state: Any, params: Any, drive: "torch.Tensor") -> "torch.Tensor":
            return params.nodes.time_const

        def write_state_velocity(self, vel: Any, state: Any, params: Any, target_sum: Any,
                                 x_t: "torch.Tensor", dt: float, **kwargs: Any) -> None:
            dt = float(dt)
            drive = params.nodes.bias + target_sum(self.currents(state, params)) + x_t
            tau = self.tau_effective(state, params, drive).clamp(min=_TAU_EPS)
            rate = (1.0 - torch.exp(-dt / tau)) / dt
            vel.nodes.activity = rate * (drive - state.nodes.activity)

    class CfCDynamics(ExactExponentialDynamics):
        """Exact integrator with an input-dependent, per-type time constant. Liquid arms.

        ``tau_eff = tau * exp(tanh(gate_v*v + gate_u*drive + gate_b))``; gates are the
        :class:`CfCGate` node parameters ``gate_v``, ``gate_u``, ``gate_b``.
        """

        def tau_effective(self, state: Any, params: Any, drive: "torch.Tensor") -> "torch.Tensor":
            n = params.nodes
            try:
                gv, gu, gb = n.gate_v, n.gate_u, n.gate_b
            except KeyError as exc:
                raise KeyError(
                    f"{CFC_DYNAMICS_NAME} needs node_config gate_v/gate_u/gate_b; build the "
                    "config with flyvis_seam.flyvis_arm_config(liquid_tau=True)") from exc
            z = gv * state.nodes.activity + gu * drive + gb
            return n.time_const * torch.exp(torch.tanh(z))

    #: (flyvis base, class) for every adapter. Module-level strong references.
    _ADAPTERS: Tuple[Tuple[type, type], ...] = (
        (InitialDistribution, SoftplusValue),
        (Parameter, CfCGate),
        (Parameter, InertFiller),
        (Parameter, DaleSynapseSign),
        (NetworkDynamics, ExactExponentialDynamics),
        (NetworkDynamics, CfCDynamics),
    )
else:
    _ADAPTERS = ()


SEAM_STATUS: Dict[str, Any] = {"registered": False, "flyvis_available": False, "classes": [], "error": _FLYVIS_IMPORT_ERROR}


def _subclass_tree(base: type) -> List[type]:
    seen, out, stack = set(), [], list(base.__subclasses__())
    while stack:
        c = stack.pop()
        if id(c) in seen:
            continue
        seen.add(id(c))
        out.append(c)
        stack.extend(c.__subclasses__())
    return out


def adapter_conflicts() -> List[str]:
    """Problems that would make flyvis dispatch to something other than this module's classes."""
    from flyvis.utils.class_utils import find_subclass

    problems = []
    for base, cls in _ADAPTERS:
        same = [c for c in _subclass_tree(base) if c.__qualname__ == cls.__qualname__]
        if len(same) != 1:
            problems.append(f"{cls.__qualname__}: {len(same)} live classes under {base.__name__} "
                            f"(modules {[c.__module__ for c in same]})")
        if find_subclass(base, cls.__qualname__) is not cls:
            problems.append(f"find_subclass({base.__name__}, {cls.__qualname__!r}) is not {__name__}.{cls.__qualname__}")
    return problems


def register_flyvis_adapters() -> Dict[str, Any]:
    """Confirm the adapters are the classes flyvis will dispatch to. Returns :data:`SEAM_STATUS`.

    Definition happens at import. This refuses (``registered=False``) when flyvis is
    missing or when a stale class from an earlier import/reload would win dispatch -- the
    fix for the latter is Runtime > Restart session, then import once.
    """
    global SEAM_STATUS
    if _FLYVIS_IMPORT_ERROR is not None:
        SEAM_STATUS = {"registered": False, "flyvis_available": False, "classes": [], "error": _FLYVIS_IMPORT_ERROR}
        return SEAM_STATUS
    problems = adapter_conflicts()
    if problems:
        gc.collect()
        problems = adapter_conflicts()
    SEAM_STATUS = {
        "registered": not problems,
        "flyvis_available": True,
        "classes": [c.__qualname__ for _, c in _ADAPTERS],
        "error": None if not problems else (
            "stale adapter classes are still alive (module reloaded?): " + "; ".join(problems)
            + " -- restart the Python session and import flyvis_seam once."),
    }
    return SEAM_STATUS


# =============================================================================
# helpers for analysis code
# =============================================================================


def trainable_parameter_counts(module: Any) -> Dict[str, int]:
    """``{name: numel}`` for every parameter with ``requires_grad``."""
    return {k: int(p.numel()) for k, p in module.named_parameters() if p.requires_grad}


def semantic_from_state_dict(state_dict: Mapping[str, Any], network_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert checkpoint RAW parameters to semantic values using the network config.

    flyvis checkpoints and `Ensemble.parameters()` hold raw values; for a SoftplusValue
    parameter the raw value is not the magnitude (audit gap on I20). Diagnostics such as
    "distance moved" or "magnitudes near zero" must use this, or `Network.param_api()`.
    """
    import torch.nn.functional as F

    out: Dict[str, Any] = {}
    for section, prefix in (("node_config", "nodes_"), ("edge_config", "edges_")):
        for name, pc in dict(network_config.get(section, {})).items():
            key = prefix + name
            if key not in state_dict:
                continue
            raw = state_dict[key]
            out[key] = F.softplus(raw) if pc.get("initial_dist") == SOFTPLUS_VALUE_NAME else raw
    return out


# =============================================================================
# acceptance test -- builds real networks; run in the runtime's own device regime
# =============================================================================


def verify_seam(verbose: bool = True, *, extent: int = 15, dt: float = 0.02, t_pre: float = 0.5,
                n_frames: int = 19, check_roundtrip: bool = True) -> Dict[str, Any]:
    """Build the arms inside flyvis and check every property Stage 2 depends on.

    Networks: A (baseline), D (Dale only), B (Dale + CfC), at ``extent`` (15 = the full
    45,669-neuron model). Returns ``{"ok", "steps", "numbers", "error"}``; ``ok`` is True
    only if every step passed. Never passes without flyvis.
    """
    report: Dict[str, Any] = {"ok": False, "steps": [], "numbers": {}, "error": None}
    nums = report["numbers"]

    def step(name: str, ok: bool, detail: str = "") -> bool:
        report["steps"].append({"step": name, "ok": bool(ok), "detail": detail})
        if verbose:
            print(f"  [{'OK ' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
        return bool(ok)

    try:
        status = register_flyvis_adapters()
        if not step("adapters are the classes flyvis dispatches to", status["registered"],
                    status["error"] or ", ".join(status["classes"])):
            report["error"] = status["error"]
            return report

        import warnings

        from datamate import namespacify
        from flyvis.network.network import Network
        from flyvis.utils.class_utils import forward_subclass

        device = torch.empty(0).device
        nums["device"] = str(device)
        nums["extent"] = int(extent)

        kw = {name: flyvis_network_kwargs(extent=extent, **ARM_PRESETS[name]) for name in ("A", "dale_only", "B")}
        cA, cD, cB = kw["A"], kw["dale_only"], kw["B"]
        s_ab = assert_configs_share_substrate(cA, cB)
        s_ad = assert_configs_share_substrate(cA, cD)
        s_db = assert_configs_share_substrate(cD, cB)
        step("merged configs share the substrate",
             s_ab["axes"] == ["dale", "liquid_tau"] and s_ad["axes"] == ["dale"] and s_db["axes"] == ["liquid_tau"],
             f"A-B {s_ab['axes']}, A-D {s_ad['axes']}, D-B {s_db['axes']}")

        def build(cfg: Dict[str, Any]) -> Any:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                net = Network(**copy.deepcopy(cfg))
            fallback = [str(w.message) for w in caught if "nrecognized" in str(w.message)]
            if fallback:
                raise AssertionError(f"flyvis fell back to a base class: {fallback}")
            return net

        A, D, B = build(cA), build(cD), build(cB)

        # -- identity, not names ---------------------------------------------
        ident = (
            type(A.dynamics) is ExactExponentialDynamics and type(D.dynamics) is ExactExponentialDynamics
            and type(B.dynamics) is CfCDynamics
            and all(type(n.edge_params.syn_strength.parameter) is SoftplusValue for n in (A, D, B))
            and type(A.edge_params.sign) is SynapseSign
            and type(D.edge_params.sign) is DaleSynapseSign and type(B.edge_params.sign) is DaleSynapseSign
            and all(type(B.node_params[k]) is CfCGate for k in GATE_KEYS)
            and all(type(n.node_params[k]) is InertFiller for n in (A, D) for k in FILLER_KEYS)
        )
        step("built classes are this module's adapters (identity)", ident,
             f"A={type(A.dynamics).__name__}, B={type(B.dynamics).__name__}, sign B={type(B.edge_params.sign).__name__}")

        clamp_ok = all(
            n.clamp_config["edges_syn_strength"] is None
            and list(n.clamp_config["nodes_time_const"]) == [TAU_MIN, TAU_MAX]
            for n in (A, D, B))
        step("clamp: syn_strength None, time_const [TAU_MIN, TAU_MAX]", clamp_ok,
             f"{A.clamp_config['edges_syn_strength']!r}, {list(A.clamp_config['nodes_time_const'])}")

        # -- parameter accounting ------------------------------------------------
        n_types = len(A.node_params.bias.keys)
        n_pairs = len(A.edge_params.syn_strength.keys)
        base = 2 * n_types + n_pairs
        want = base + 3 * n_types
        counts = {k: sum(trainable_parameter_counts(n).values()) for k, n in (("A", A), ("dale_only", D), ("B", B))}
        nums.update(n_types=n_types, n_pairs=n_pairs, stock_trainable=base, trainable=counts)
        ok_counts = all(v == want for v in counts.values())
        if extent == 15:
            ok_counts = ok_counts and base == 734 and want == 929
        step("free parameters matched across arms", ok_counts,
             f"{counts} (stock {base}; expected {want} each)")
        per_gate = {k: int(getattr(B, f"nodes_{k}").numel()) for k in GATE_KEYS}
        step("gates are per cell type", all(v == n_types for v in per_gate.values()), f"{per_gate}")

        # -- initialization equals stock flyvis where it should -------------------
        def stock_param(section_cfg: Dict[str, Any], name: str, net: Any) -> Any:
            pc = namespacify(copy.deepcopy(section_cfg[name]))
            return forward_subclass(Parameter, {"type": pc.type, "param_config": pc, "connectome": net.connectome})

        stock_ss = dict(cA["edge_config"]["syn_strength"], initial_dist="Value", clamp="non_negative")
        ref = stock_param({"s": stock_ss}, "s", A).semantic_values.detach()
        sem = A.edge_params.syn_strength.semantic_values.detach()
        rel = float(((sem - ref).abs() / ref.abs().clamp(min=1e-12)).max())
        nums["softplus_init_max_rel_err"] = rel
        step("SoftplusValue semantic init == flyvis Value init", rel < 1e-4, f"max rel err {rel:.2e}")

        same_init = all(
            torch.equal(getattr(A, k), getattr(n, k))
            for n in (D, B) for k in ("nodes_bias", "nodes_time_const", "edges_syn_strength", "edges_syn_count"))
        step("bias/tau/magnitude/count init identical across arms", same_init)

        # -- Dale -----------------------------------------------------------------
        stock_sign = stock_param(cA["edge_config"], "sign", A).raw_values.detach()
        plan = B.edge_params.sign.dale_plan
        nums["dale_plan"] = plan.as_dict()
        n_diff = int((A.edges_sign != B.edges_sign).sum())
        step("arm A signs == stock flyvis signs", torch.equal(A.edges_sign.detach(), stock_sign))
        step("Dale flips groups, and exactly the planned ones",
             plan.n_flipped > 0 and n_diff == plan.n_flipped and torch.equal(D.edges_sign, B.edges_sign),
             f"{n_diff} groups differ A vs B; plan {plan.n_flipped}: {', '.join(nums['dale_plan']['flipped'])}")
        per_type: Dict[str, set] = {}
        for (src, _), s in zip(B.edge_params.sign.keys, B.edges_sign.detach().cpu().tolist()):
            per_type.setdefault(src, set()).add(s)
        mixed_B = sorted(t for t, ss in per_type.items() if len(ss) > 1)
        step("one sign per source type in the Dale arms", not mixed_B, f"mixed: {mixed_B}; before: {list(plan.mixed_types)}")

        # -- dynamics ---------------------------------------------------------------
        params_A = A._param_api()
        ss_A = A.steady_state(t_pre=t_pre, dt=dt, batch_size=1, value=0.5)
        A.stimulus.zero(1, 1)
        A.stimulus.add_pre_stim(1.0)
        x0 = A.stimulus()[:, 0]
        v0 = ss_A.nodes.activity
        v1 = A._next_state(params_A, ss_A, x0, dt).nodes.activity
        u = params_A.nodes.bias + A.target_sum(A.dynamics.currents(ss_A, params_A)) + x0
        keep = torch.exp(-dt / params_A.nodes.time_const)
        closed = keep * v0 + (1 - keep) * u
        err = float((v1 - closed).abs().max() / closed.abs().max().clamp(min=1e-12))
        nums["exact_update_rel_err"] = err
        step("one flyvis step == exact closed form e^(-dt/tau) v + (1-e^(-dt/tau)) u", err < 1e-5, f"rel err {err:.2e}")
        del params_A, ss_A, v0, v1, u, keep, closed

        def run(net: Any) -> Any:
            ss = net.steady_state(t_pre=t_pre, dt=dt, batch_size=1, value=0.5)
            net.stimulus.zero(1, n_frames)
            net.stimulus.add_pre_stim(1.0)
            return net(net.stimulus(), dt, state=ss)

        ids_before = {id(p) for p in B.parameters()}
        opts = {k: torch.optim.Adam(n.parameters(), lr=1e-3) for k, n in (("A", A), ("dale_only", D), ("B", B))}
        acts = {}
        for k, n in (("A", A), ("dale_only", D), ("B", B)):
            opts[k].zero_grad()
            acts[k] = run(n)
        step("no parameters created lazily during forward (optimizer built first holds all)",
             ids_before == {id(p) for p in B.parameters()})
        vmax = {k: float(a.detach().abs().max()) for k, a in acts.items()}
        nums["max_abs_activity"] = vmax
        step("activity finite and bounded (steady state + luminance step)",
             all(torch.isfinite(a).all() for a in acts.values()) and max(vmax.values()) < 1e3, f"max|v| {vmax}")
        d_AD = float((acts["A"] - acts["dale_only"]).detach().abs().max())
        d_DB = float((acts["dale_only"] - acts["B"]).detach().abs().max())
        nums.update(max_diff_A_vs_dale_only=d_AD, max_diff_dale_only_vs_B=d_DB)
        step("Dale changes activity (A vs Dale-only)", d_AD > 0, f"max|diff| {d_AD:.3e}")
        # 1e-5, not 0: CUDA scatter_add_ need not be bit-reproducible across two networks
        step("zero gates == no gates (Dale-only vs B at init)", d_DB <= 1e-5,
             f"max|diff| {d_DB:.3e}{' (bit-identical)' if d_DB == 0 else ''}")

        # -- gradients, flyvis's order: backward(retain) -> step -> penalty backward ---
        grads: Dict[str, Dict[str, float]] = {}
        try:
            for k, n in (("A", A), ("B", B)):
                loss = acts[k].pow(2).mean()
                loss.backward(retain_graph=True)
                grads[k] = {name: (None if p.grad is None else float(p.grad.norm()))
                            for name, p in n.named_parameters() if p.requires_grad}
                before = {name: p.detach().clone() for name, p in n.named_parameters() if p.requires_grad}
                opts[k].step()
                acts[k][:, n_frames // 4:].mean().pow(2).backward()   # the activity-penalty backward
                n.clamp()
                grads[k]["_moved"] = [name for name, p in n.named_parameters()
                                      if p.requires_grad and not torch.equal(before[name], p.detach())]
            step("flyvis order backward/step/penalty-backward runs", True)
        except RuntimeError as exc:
            step("flyvis order backward/step/penalty-backward runs", False, f"{type(exc).__name__}: {exc}")
            raise
        nums["grad_norms"] = grads
        gate_g = {g: grads["B"][f"nodes_{g}"] for g in GATE_KEYS}
        step("gate gradients non-zero and finite", all(v is not None and 0 < v < float("inf") for v in gate_g.values()),
             f"{gate_g}")
        core_ok = all(grads[k][p] is not None and 0 < grads[k][p] < float("inf")
                      for k in ("A", "B") for p in ("nodes_bias", "nodes_time_const", "edges_syn_strength"))
        step("bias/tau/magnitude gradients non-zero and finite", core_ok)
        filler_ok = all(grads["A"][f"nodes_{f}"] is None for f in FILLER_KEYS) and \
            not any(m.startswith("nodes_filler") for m in grads["A"]["_moved"])
        step("filler is inert (grad None, unchanged after a step)", filler_ok)
        step("gates move after one step", all(f"nodes_{g}" in grads["B"]["_moved"] for g in GATE_KEYS))
        del acts, opts

        # -- checkpoint round trip -------------------------------------------------
        if check_roundtrip:
            sd = {k: v.detach().clone() for k, v in B.state_dict().items()}
            del D
            gc.collect()
            B2 = build(cB)
            missing = B2.load_state_dict(sd, strict=True)
            with torch.no_grad():
                d_rt = float((run(B) - run(B2)).abs().max())
            nums["roundtrip_max_diff"] = d_rt
            step("state dict round trip (strict) reproduces arm B", d_rt <= 1e-5 and not missing.missing_keys,
                 f"{len(sd)} tensors, max|diff| {d_rt:.1e}")
            del B2

        report["ok"] = all(s["ok"] for s in report["steps"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        step("verify_seam", False, report["error"])
    return report


# =============================================================================
# self-test -- pure arithmetic, Dale planning and config logic; no flyvis required
# =============================================================================


def _main() -> int:
    import time

    t0 = time.perf_counter()
    n = 0
    line = "=" * 78

    def ok(msg: str) -> None:
        nonlocal n
        n += 1
        print(f"  [PASS] {msg}")

    def raises(fn: Any, exc: type, needle: str) -> str:
        try:
            fn()
        except exc as e:
            assert needle in str(e), (needle, str(e))
            return str(e)
        raise AssertionError(f"expected {exc.__name__} containing {needle!r}")

    print(line)
    print("flyvis_seam.py self-test (arithmetic, Dale planning, configs; no flyvis needed)")
    print(line)

    print("\n-- velocity coefficients --")
    dt, tau = 0.02, 0.05
    ke, kx = retention(dt, euler_rate(dt, tau)), retention(dt, exact_rate(dt, tau))
    assert abs(ke - 0.600000) < 1e-9 and abs(kx - 0.670320) < 1e-6, (ke, kx)
    ok(f"at dt={dt}, tau={tau}: euler retention {ke:.6f}, exact {kx:.6f}")
    rel = abs(kx - ke) / ke
    assert 0.117 < rel < 0.118, rel
    ok(f"relative difference {rel:.2%}")
    assert abs(kx - math.exp(-dt / tau)) < 1e-12
    ok("exact retention equals exp(-dt/tau)")
    errs = []
    for big in (1.0, 10.0, 100.0):
        err = abs(exact_rate(dt, big) * big - 1.0)
        assert err < dt / big, (big, err)
        errs.append(err)
    assert errs[0] > errs[1] > errs[2]
    ok(f"exact_rate -> 1/tau as dt/tau -> 0; rel err {errs[0]:.2e} -> {errs[2]:.2e}")
    rows = rate_comparison()
    assert any(r["euler_gradient_dead"] for r in rows) and all(exact_rate(dt, r["tau"]) > 0 for r in rows)
    ok("Euler's guard is dead below dt; the exact rate stays positive there")
    # TAU_MIN keeps the exact rate's derivative usable at the training dt
    d_rate = math.exp(-dt / TAU_MIN) * (1.0 / TAU_MIN ** 2)   # |d rate / d tau| = e^(-dt/tau)/tau^2
    assert d_rate > 1.0, d_rate
    ok(f"|d rate/d tau| at TAU_MIN={TAU_MIN}: {d_rate:.2f} (non-negligible)")

    for y in (1e-5, 3e-4, 5e-3, 0.7, 25.0):
        x = inverse_softplus(y)
        back = math.log1p(math.exp(x)) if x < 30 else x
        assert abs(back - y) <= 1e-9 * max(1.0, y), (y, back)
    raises(lambda: inverse_softplus(0.0), ValueError, "y > 0")
    ok("inverse_softplus round-trips 1e-5 .. 25 and refuses y <= 0")

    print("\n-- Dale sign plan (group level) --")
    keys = [("Am", "L1"), ("Am", "T1"), ("Am", "Mi1"), ("L1", "Mi1"), ("L1", "L5"), ("Mi9", "T4a"), ("R8", "Mi1"), ("R8", "Mi4")]
    signs = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
    raises(lambda: DaleSignPlan.from_groups(keys, signs, {}), ValueError, "emits both signs")
    ok("a mixed-sign type with no published sign is REFUSED under scope 'all'")
    p = DaleSignPlan.from_groups(keys, signs, {"Am": -1, "R8": -1}, group_weight=[1, 3, 2, 5, 5, 4, 2, 2])
    assert p.flipped == [("Am", "L1"), ("Am", "T1"), ("R8", "Mi1")], p.flipped
    assert p.mixed_types == ("Am", "R8") and p.mixed_after() == []
    assert abs(p.pct_weight_flipped - 100 * 6 / 24) < 1e-9
    ok(f"explicit signs flip exactly {p.n_flipped} groups, {p.pct_weight_flipped:.1f}% of weight, no mixed types left")
    c = DaleSignPlan.from_groups(keys, signs, {"Am": -1, "R8": -1}, scope="unambiguous_only")
    assert c.n_flipped == 0 and c.mixed_after() == ["Am", "R8"]
    ok("scope 'unambiguous_only' changes nothing when unambiguous types already obey Dale")
    raises(lambda: DaleSignPlan.from_groups(keys, signs, {"Am": -1, "R8": -1, "L1": -1}), ValueError, "refusing")
    ok("a published sign contradicting a consistent type is REFUSED")
    raises(lambda: DaleSignPlan.from_groups(keys, signs, {"Am": -1, "R8": -1}, scope="some"), ValueError, "dale_scope")
    ok("unknown scope is REFUSED")

    print("\n-- arm configs --")
    signs_pub = {"Am": -1.0, "C2": -1.0, "C3": -1.0, "R8": -1.0}
    mk = lambda **k: flyvis_arm_config(dale_signs=signs_pub if k.get("dale") else None, **k)
    a, b = mk(dale=False, liquid_tau=False), mk(dale=True, liquid_tau=True)
    d, l = mk(dale=True, liquid_tau=False), mk(dale=False, liquid_tau=True)
    for cfg in (a, b):
        ss = cfg["edge_config"]["syn_strength"]
        assert ss["initial_dist"] == SOFTPLUS_VALUE_NAME and ss["clamp"] is None
        assert cfg["node_config"]["time_const"]["clamp"] == [TAU_MIN, TAU_MAX]
    ok("both arms: SoftplusValue, clamp None, tau clamp [TAU_MIN, TAU_MAX]")
    assert a["dynamics"]["type"] == EXACT_DYNAMICS_NAME and b["dynamics"]["type"] == CFC_DYNAMICS_NAME
    assert b["edge_config"]["sign"]["type"] == DALE_SIGN_NAME and a["edge_config"]["sign"]["type"] == "SynapseSign"
    assert "_dale_override" not in str(b)
    ok("treatments are classes (CfCDynamics, DaleSynapseSign), not ignorable flags")
    assert set(FILLER_KEYS) <= set(a["node_config"]) and set(GATE_KEYS) <= set(b["node_config"])
    assert all(b["node_config"][g]["groupby"] == ["type"] and b["node_config"][g]["value"] == 0.0 for g in GATE_KEYS)
    ok("gates per type, zero init; filler of the same shape in the non-liquid arm")

    r = assert_configs_share_substrate(a, b)
    assert r["axes"] == ["dale", "liquid_tau"], r
    assert assert_configs_share_substrate(a, d)["axes"] == ["dale"]
    assert assert_configs_share_substrate(a, l)["axes"] == ["liquid_tau"]
    assert assert_configs_share_substrate(d, b)["axes"] == ["liquid_tau"]
    ok("A-B differ in dale+liquid_tau; single-axis arms differ in exactly one")

    bad = copy.deepcopy(b)
    bad["edge_config"]["syn_strength"]["clamp"] = "non_negative"
    raises(lambda: assert_configs_share_substrate(a, bad), AssertionError, "syn_strength.clamp")
    stock = copy.deepcopy(a)
    stock["dynamics"]["type"] = "PPNeuronIGRSynapses"
    raises(lambda: assert_configs_share_substrate(stock, stock), AssertionError, "dynamics.type")
    ok("a reintroduced clamp, or two STOCK configs, fail the substrate check")
    bad2 = copy.deepcopy(a)
    bad2["dynamics"]["activation"]["type"] = "tanh"
    raises(lambda: assert_configs_share_substrate(a, bad2), AssertionError, "activation")
    bad3 = copy.deepcopy(b)
    bad3["edge_config"]["sign"]["type"] = "SynapseSign"
    raises(lambda: assert_configs_share_substrate(a, bad3), AssertionError, "dale_signs present")
    bad4 = copy.deepcopy(b)
    del bad4["node_config"]["gate_u"]
    raises(lambda: assert_configs_share_substrate(a, bad4), AssertionError, "gates present")
    unmatched = flyvis_arm_config(dale=False, liquid_tau=False, param_matched=False)
    raises(lambda: assert_configs_share_substrate(unmatched, b), AssertionError, "parameter matching")
    ok("differing activation, label-only Dale, missing gate, unmatched params: all REFUSED")
    raises(lambda: flyvis_arm_config(dale=False, liquid_tau=False, tying=["source_type"]), ValueError, "tying")
    raises(lambda: flyvis_arm_config(dale=False, liquid_tau=False, tau_init=1e-4), ValueError, "tau_init")
    ok("invalid tying and tau_init outside the clamp are REFUSED")

    print("\n-- merging --")
    defaults = {
        "node_config": {"bias": {"type": "RestingPotential", "seed": 0},
                        "time_const": {"type": "TimeConstant", "groupby": ["type"], "initial_dist": "Value",
                                       "value": 0.05, "requires_grad": True}},
        "edge_config": {"sign": {"type": "SynapseSign", "initial_dist": "Value", "requires_grad": False,
                                 "groupby": ["source_type", "target_type"]},
                        "syn_count": {"type": "SynapseCount", "initial_dist": "Lognormal"},
                        "syn_strength": {"type": "SynapseCountScaling", "initial_dist": "Value",
                                         "clamp": "non_negative", "penalize": {"function": "weight_decay"}}},
        "dynamics": {"type": "PPNeuronIGRSynapses", "activation": {"type": "relu"}},
    }
    mb = {k: deep_merge(defaults[k], b[k]) for k in ("node_config", "edge_config")}
    mb["dynamics"] = copy.deepcopy(b["dynamics"])
    assert "syn_count" in mb["edge_config"] and "bias" in mb["node_config"]
    assert mb["edge_config"]["syn_strength"]["clamp"] is None
    assert mb["edge_config"]["syn_strength"]["penalize"] == {"function": "weight_decay"}
    ok("merge keeps syn_count, bias and penalize; overrides clamp to None")
    back = deep_merge(mb["edge_config"], a["edge_config"])
    assert "dale_signs" not in back["sign"], back["sign"]
    ok("merging arm A over an arm-B config drops dale_signs (type change replaces the entry)")
    assert defaults["edge_config"]["syn_strength"]["clamp"] == "non_negative"
    ok("merge does not mutate its inputs")

    print("\n-- flyvis --")
    st = register_flyvis_adapters()
    if st["flyvis_available"]:
        ok(f"flyvis present; registered={st['registered']}. Run verify_seam() for the live check")
    else:
        assert st["registered"] is False
        ok(f"flyvis absent: {str(st['error'])[:60]}")
        rep = verify_seam(verbose=False)
        assert rep["ok"] is False
        ok("verify_seam() is NOT ok without flyvis")

    print("\n" + line)
    print(f"ALL {n} CHECKS PASSED in {time.perf_counter() - t0:.2f} s")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
