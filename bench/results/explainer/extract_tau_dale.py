"""Extract two small sets of real numbers from flyvis's released models + connectome, for an
explainer figure:

  (a) per-cell-type time constant tau (seconds), for models 000/001/002, plus the initial
      (untrained) config value.
  (b) Dale-flip group stats (edges, synapse count, connectome sign) for the 10 published-sign
      flip groups, cross-checked against src/flyvis_seam.py's DaleSignPlan / published signs
      and against FINDINGS.md's "10 of 604 groups (8,590 edges)".

Run with the project's CPU-torch venv, FLYVIS_ROOT_DIR pointed at the cached connectome +
released models, and the datamate Windows patch applied (see module docstring requirements
in the task). Writes explainer/tau_dale.json next to this script.
"""
import json
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])      # repository root
sys.path[:0] = [ROOT + "/bench", ROOT + "/src"]

# Must be set before importing flyvis: a root holding the connectome cache and results/flow/0000.
os.environ.setdefault("FLYVIS_ROOT_DIR", ROOT + "/flyvis_data")

import m2_released  # noqa: E402 (bench/m2_released.py)

m2_released.patch_datamate_for_windows()

import numpy as np  # noqa: E402
from flyvis import NetworkView  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
MODEL_IDS = ["000", "001", "002"]

CKPT_KW = {"validation_subdir": "validation", "loss_file_name": "loss"}

DALE_PAIRS = [
    ("R8", "Mi1"), ("R8", "Mi4"), ("R8", "Mi9"), ("R8", "Mi15"),
    ("Am", "L1"), ("Am", "T1"),
    ("C2", "L1"), ("C2", "L2"),
    ("C3", "L1"), ("C3", "L2"),
]


def log(*a):
    print(*a, flush=True)


def extract_tau(model_id):
    log(f"[tau] loading flow/0000/{model_id} ...")
    nv = NetworkView(f"flow/0000/{model_id}", best_checkpoint_fn_kwargs=CKPT_KW)
    net = nv.init_network()

    tau_param = net.node_params["time_const"]
    keys = list(tau_param.keys)  # cell-type name per entry, in grouped-node order
    values = tau_param.semantic_values.detach().cpu().numpy().reshape(-1)
    assert len(keys) == len(values), (len(keys), len(values))

    tau_by_type = {str(k): float(v) for k, v in zip(keys, values)}

    # cross-check: the 65 type names from the connectome match the tau keys (as a set)
    ct = [c.decode() if isinstance(c, bytes) else str(c) for c in net.connectome.unique_cell_types[:]]
    n_types_connectome = len(ct)
    keys_match_connectome = set(str(k) for k in keys) == set(ct)

    # per-type resting potential / bias, if trivially available
    bias_by_type = None
    try:
        bias_param = net.node_params.get("bias", None)
        if bias_param is None:
            for cand in ("bias", "resting_potential"):
                if cand in net.node_params:
                    bias_param = net.node_params[cand]
                    break
        if bias_param is not None:
            bkeys = list(bias_param.keys)
            bvals = bias_param.semantic_values.detach().cpu().numpy().reshape(-1)
            bias_by_type = {str(k): float(v) for k, v in zip(bkeys, bvals)}
    except Exception as e:
        log(f"  [bias] not available for {model_id}: {e!r}")

    # initial (untrained, config) tau value -- same for every model (config, not learned per type
    # here: flyvis inits every type to the same scalar `value`, then trains per-type from there)
    tau_init_from_config = None
    try:
        tau_init_from_config = float(net.config.node_config.time_const.value)
    except Exception as e:
        log(f"  [tau_init via net.config] failed: {e!r}")
    tau_init_from_dir_config = None
    try:
        tau_init_from_dir_config = float(nv.dir.config.network.node_config.time_const.value)
    except Exception as e:
        log(f"  [tau_init via nv.dir.config] failed: {e!r}")

    log(f"[tau] {model_id}: {len(tau_by_type)} types, "
        f"tau range [{min(tau_by_type.values()):.6g}, {max(tau_by_type.values()):.6g}]")

    return {
        "tau_by_type": tau_by_type,
        "bias_by_type": bias_by_type,
        "n_types_connectome": n_types_connectome,
        "keys_match_connectome": keys_match_connectome,
        "tau_init_from_config": tau_init_from_config,
        "tau_init_from_dir_config": tau_init_from_dir_config,
        "flyvis_version": getattr(__import__("flyvis"), "__version__", None),
    }, net


def extract_dale(net):
    edges_df = net.connectome.edges.to_df()
    log(f"[dale] edges_df columns: {list(edges_df.columns)}")
    log(f"[dale] edges_df shape: {edges_df.shape}")

    # normalize byte-string columns to str, if needed
    for col in ("source_type", "target_type"):
        if edges_df[col].dtype == object and len(edges_df) and isinstance(edges_df[col].iloc[0], bytes):
            edges_df[col] = edges_df[col].str.decode("utf-8")

    groups = []
    total_edges = 0
    for src, tgt in DALE_PAIRS:
        sub = edges_df[(edges_df["source_type"] == src) & (edges_df["target_type"] == tgt)]
        n_edges = int(len(sub))
        n_syn = float(sub["n_syn"].sum()) if n_edges else 0.0
        signs = sorted(set(sub["sign"].unique().tolist())) if n_edges else []
        mixed = len(signs) > 1
        connectome_sign = int(signs[0]) if len(signs) == 1 else None
        groups.append({
            "source": src,
            "target": tgt,
            "n_edges": n_edges,
            "n_syn": n_syn,
            "connectome_sign": connectome_sign,
            "connectome_signs_seen": signs,
            "mixed_sign": mixed,
        })
        total_edges += n_edges

    n_edges_total_all_pairs_in_connectome = int(len(edges_df))
    n_groups_total = int(
        edges_df.groupby(["source_type", "target_type"]).ngroups
    )

    return {
        "groups": groups,
        "n_edges_total": total_edges,
        "n_edges_all_connectome": n_edges_total_all_pairs_in_connectome,
        "n_groups_total_connectome": n_groups_total,
    }


def cross_check_flyvis_seam(edges_df=None):
    """Cross-check the flip-group list and published signs against src/flyvis_seam.py /
    src/dale_typed_cfc.py, if importable (pure-python, no torch needed per their self-tests).

    If `edges_df` is given, also runs flyvis_seam.DaleSignPlan over ALL (source_type,
    target_type) groups in the real connectome (not just the 10 pairs we assumed) and
    reports which groups it flips, so the 10-pair list is verified programmatically
    rather than assumed.
    """
    out = {"available": False}
    try:
        import dale_typed_cfc as dtc
        pub = dtc.published_explicit_signs()
        out["published_signs"] = {k: float(v) for k, v in pub.items()}
        out["available"] = True
    except Exception as e:
        out["error"] = repr(e)
        return out

    if edges_df is None:
        return out

    try:
        from flyvis_seam import DaleSignPlan

        grouped = edges_df.groupby(["source_type", "target_type"], as_index=False, sort=False).first()
        keys = list(zip(grouped["source_type"].tolist(), grouped["target_type"].tolist()))
        signs = [float(s) for s in grouped["sign"].tolist()]
        plan = DaleSignPlan.from_groups(keys, signs, pub, scope="all")
        flipped_pairs = sorted(f"{a}->{b}" for a, b in plan.flipped)
        expected_pairs = sorted(f"{s}->{t}" for s, t in DALE_PAIRS)
        out["plan_n_groups_total"] = len(keys)
        out["plan_n_flipped"] = plan.n_flipped
        out["plan_flipped_pairs"] = flipped_pairs
        out["plan_matches_task_pairs"] = flipped_pairs == expected_pairs
        out["plan_mixed_types"] = list(plan.mixed_types)
    except Exception as e:
        out["plan_error"] = repr(e)

    return out


def main():
    result = {"tau_s": {}, "per_model_extra": {}, "tau_init_s": None, "dale_groups": [], "n_edges_total": None}

    net_for_dale = None
    tau_inits = []
    for mid in MODEL_IDS:
        tau_info, net = extract_tau(mid)
        result["tau_s"][mid] = tau_info["tau_by_type"]
        result["per_model_extra"][mid] = {
            "bias_by_type": tau_info["bias_by_type"],
            "n_types_connectome": tau_info["n_types_connectome"],
            "keys_match_connectome": tau_info["keys_match_connectome"],
            "tau_init_from_config": tau_info["tau_init_from_config"],
            "tau_init_from_dir_config": tau_info["tau_init_from_dir_config"],
            "flyvis_version": tau_info["flyvis_version"],
        }
        if tau_info["tau_init_from_config"] is not None:
            tau_inits.append(tau_info["tau_init_from_config"])
        if net_for_dale is None:
            net_for_dale = net  # connectome is shared across models (same connectome cache); use model 000's

    if tau_inits:
        uniq = sorted(set(round(x, 12) for x in tau_inits))
        result["tau_init_s"] = tau_inits[0]
        result["tau_init_consistent_across_models"] = len(uniq) == 1
        result["tau_init_all_values"] = tau_inits
    else:
        result["tau_init_consistent_across_models"] = None
        result["tau_init_all_values"] = []

    dale_info = extract_dale(net_for_dale)
    full_edges_df = net_for_dale.connectome.edges.to_df()
    for col in ("source_type", "target_type"):
        if full_edges_df[col].dtype == object and len(full_edges_df) and isinstance(full_edges_df[col].iloc[0], bytes):
            full_edges_df[col] = full_edges_df[col].str.decode("utf-8")
    seam_check = cross_check_flyvis_seam(full_edges_df)
    pub_signs = seam_check.get("published_signs", {})

    dale_groups_out = []
    for g in dale_info["groups"]:
        dale_sign = pub_signs.get(g["source"])
        dale_sign_int = int(dale_sign) if dale_sign is not None else None
        dale_groups_out.append({
            "source": g["source"],
            "target": g["target"],
            "n_edges": g["n_edges"],
            "n_syn": g["n_syn"],
            "connectome_sign": g["connectome_sign"],
            "dale_sign": dale_sign_int,
        })

    result["dale_groups"] = dale_groups_out
    result["n_edges_total"] = dale_info["n_edges_total"]
    result["dale_extra"] = {
        "groups_raw": dale_info["groups"],
        "n_edges_all_connectome": dale_info["n_edges_all_connectome"],
        "n_groups_total_connectome": dale_info["n_groups_total_connectome"],
        "findings_md_n_edges_total": 8590,
        "findings_md_n_groups_flipped": 10,
        "findings_md_n_groups_total": 604,
        "matches_findings_md_edges": dale_info["n_edges_total"] == 8590,
        "matches_findings_md_groups_total": dale_info["n_groups_total_connectome"] == 604,
        "flyvis_seam_cross_check": seam_check,
    }

    out_path = OUT_DIR / "tau_dale.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"[main] wrote {out_path}")

    # Console summary
    log("\n=== SUMMARY ===")
    for mid in MODEL_IDS:
        taus = result["tau_s"][mid]
        log(f"model {mid}: {len(taus)} types; e.g. T4c={taus.get('T4c')}, R1={taus.get('R1')}")
    log(f"tau_init_s: {result['tau_init_s']} (consistent across models: {result['tau_init_consistent_across_models']})")
    log(f"n_edges_total over 10 pairs: {result['n_edges_total']} "
        f"(FINDINGS.md: 8590; match: {result['dale_extra']['matches_findings_md_edges']})")
    log(f"n_groups_total_connectome: {dale_info['n_groups_total_connectome']} "
        f"(FINDINGS.md: 604; match: {result['dale_extra']['matches_findings_md_groups_total']})")
    for g in dale_groups_out:
        log(f"  {g['source']}->{g['target']}: n_edges={g['n_edges']} n_syn={g['n_syn']} "
            f"connectome_sign={g['connectome_sign']} dale_sign={g['dale_sign']}")
    log(f"flyvis_seam.DaleSignPlan cross-check: plan_matches_task_pairs="
        f"{seam_check.get('plan_matches_task_pairs')}, plan_n_flipped={seam_check.get('plan_n_flipped')}, "
        f"plan_n_groups_total={seam_check.get('plan_n_groups_total')}")


if __name__ == "__main__":
    main()
