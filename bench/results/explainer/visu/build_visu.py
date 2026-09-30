"""Build the Portuguese VISU explorer from its template + the measured data.

Outputs (next to this script):
  visu-anatomia.html            full document, ready for Apps Script HtmlService (index.html)
  visu-anatomia.artifact.html   the same page without <html>/<head>/<body> wrappers, for an artifact preview
"""
import json, re
from pathlib import Path

SP = Path(__file__).resolve().parent
EX = next(d for d in (SP, SP / "explainer", SP.parent) if (d / "fp_data.json").exists())   # measured data
data = {}
if (EX / "fp_data.json").exists():
    data["fp"] = json.loads((EX / "fp_data.json").read_text())
if (EX / "tuning_000.json").exists():
    t = json.loads((EX / "tuning_000.json").read_text())
    keep = ["T4a", "T4b", "T4c", "T4d"]
    idx = [i for i, s in enumerate(t["time_s_subsampled"]) if -0.2 <= s <= 1.4][::2]
    sel = set(idx)
    data["tuning"] = {"model": t["model"], "angles": [int(a) for a in t["angles"]],
                      "time_s_subsampled": [round(t["time_s_subsampled"][i], 3) for i in idx],
                      "traces": {k: [[round(v, 4) for j, v in enumerate(row) if j in sel] for row in t["traces"][k]] for k in keep},
                      "peak": {k: t["peak"][k] for k in keep}, "integral": {k: t["integral"][k] for k in keep}}
if (EX / "tau_dale.json").exists():
    data["tau"] = json.loads((EX / "tau_dale.json").read_text())
if (EX / "sintel_clip.json").exists():
    s = json.loads((EX / "sintel_clip.json").read_text())
    data["sintel"] = {"sequence": s["sequence"], "frames": s["frames"], "uv": s["uv"], "lum": s["lum"]}

tpl = (SP / "visu-anatomia.template.html").read_text(encoding="utf-8")
assert "/*__DATA__*/{}" in tpl
full = tpl.replace("/*__DATA__*/{}", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
(SP / "visu-anatomia.html").write_text(full, encoding="utf-8")

# artifact variant: drop the document wrappers and the meta tags (the publish skeleton adds its own)
inner = re.sub(r"^\s*<!DOCTYPE html>\s*<html[^>]*>\s*<head>\s*", "", full, flags=re.S)
inner = re.sub(r'<meta [^>]*>\s*', "", inner)
inner = inner.replace("</head>\n<body>\n", "").replace("</body>\n</html>\n", "")
(SP / "visu-anatomia.artifact.html").write_text(inner, encoding="utf-8")
print("built:", {k: len(json.dumps(v)) for k, v in data.items()}, "| full", len(full), "bytes | artifact", len(inner), "bytes")
