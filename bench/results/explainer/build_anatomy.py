"""Inject the measured data into the anatomy explainer template -> fly-circuit-anatomy.html"""
import json, sys
from pathlib import Path

SP = Path(__file__).resolve().parent
EX = next(d for d in (SP, SP / "explainer", SP.parent) if (d / "fp_data.json").exists())   # measured data
data = {}
fp = EX / "fp_data.json"
if fp.exists():
    data["fp"] = json.loads(fp.read_text())
tun = EX / "tuning_000.json"
if tun.exists():
    t = json.loads(tun.read_text())
    keep = ["T4a", "T4b", "T4c", "T4d"]
    # every 2nd subsampled point (dt 50 ms), T4 subtypes only, window -0.2 .. 1.4 s
    idx = [i for i, s in enumerate(t["time_s_subsampled"]) if -0.2 <= s <= 1.4][::2]
    data["tuning"] = {"model": t["model"], "angles": [int(a) for a in t["angles"]],
                      "time_s_subsampled": [round(t["time_s_subsampled"][i], 3) for i in idx],
                      "traces": {k: [[round(v, 4) for j, v in enumerate(row) if j in set(idx)] for row in t["traces"][k]] for k in keep},
                      "peak": {k: t["peak"][k] for k in keep}, "integral": {k: t["integral"][k] for k in keep}}
td = EX / "tau_dale.json"
if td.exists():
    data["tau"] = json.loads(td.read_text())
sc = EX / "sintel_clip.json"
if sc.exists():
    s = json.loads(sc.read_text())
    data["sintel"] = {"sequence": s["sequence"], "frames": s["frames"], "uv": s["uv"], "lum": s["lum"]}
tpl = (SP / "fly-circuit-anatomy.template.html").read_text(encoding="utf-8")
assert "/*__DATA__*/{}" in tpl
out = tpl.replace("/*__DATA__*/{}", json.dumps(data, separators=(",", ":")))
(SP / "fly-circuit-anatomy.html").write_text(out, encoding="utf-8")
print("built:", {k: len(json.dumps(v)) for k, v in data.items()}, "total bytes", len(out))
