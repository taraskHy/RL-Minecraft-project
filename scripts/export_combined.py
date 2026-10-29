"""Export a combined student's validation-chosen best.pt as results/combined_policy.pt/.json (+ provenance).

Weights, config and evaluation only (no optimizer or sampler state), as results/noroute_policy.pt.
Run from the repository root: python scripts/export_combined.py combined-distill
"""
import hashlib
import json
import sys
from pathlib import Path

import torch


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(run: str) -> None:
    src = Path(f"runs/{run}/checkpoints/best.pt")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    summary = json.loads(Path(f"runs/{run}/summary.json").read_text())
    assert summary.get("completed") and summary["test_eval"]["iteration"] == ck["iteration"]
    out = Path("results/combined_policy.pt")
    torch.save({"model": ck["model"], "config": ck["config"], "iteration": ck["iteration"], "eval": ck["eval"],
                "source": f"{src.as_posix()} (validation-selected; optimizer state removed)"}, out)
    js = Path(f"runs/{run}/policy_best.json")
    meta = json.loads(js.read_text())["meta"]
    assert meta["run"] == run and meta["iteration"] == ck["iteration"], meta
    Path("results/combined_policy.json").write_bytes(js.read_bytes())
    best = summary["best"]
    prov = {"source": src.as_posix(), "run": run, "iteration": ck["iteration"],
            "selection": "validation families 2/5 and ice 17 (simulator), best.pt = highest mean over "
                         "basic, ice, advanced and extra",
            "validation_macro": best["validation_success"], "validation_groups": best.get("validation_groups"),
            "test_macro": summary["test_eval"]["macro"],
            "test_groups": {g: v["all"] for g, v in summary["test_eval"]["groups"].items()},
            "contents": "weights, config and evaluation only (no optimizer state); combined_policy.json is the "
                        "bot export (parkour-policy-v1) written by train.py at the same checkpoint",
            "source_sha256": sha256(src), "export_sha256": sha256(out),
            "json_sha256": sha256(Path("results/combined_policy.json")),
            "final_policy_preserved": Path("results/final_policy.pt").exists() and Path("results/final_policy.json").exists()}
    Path("results/combined_policy_provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
