"""
Experiment result logging.

Every experiment MUST call log_run() before finishing.
If log_run() wasn't called, the experiment didn't happen (per PIPELINE.md §5).

Writes:
  results/EX_*/  <exp_id>_<git_hash>.json   — full metrics + params
  results/manifest.jsonl                    — one line per run (provenance)
  results/RESULTS.md                        — human-readable rolling summary
"""
import json
import os
import sys
import time
import subprocess


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
    except Exception:
        pass
    return "cpu"


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:
        return "unknown"


def log_run(
    exp_id: str,
    metrics: dict,
    params: dict,
    results_dir: str,
    repo_root: str = None,
) -> str:
    """
    Write a versioned JSON, append a manifest line, and update RESULTS.md.
    Returns the path to the written JSON.
    """
    os.makedirs(results_dir, exist_ok=True)

    git = _git_hash()
    rec = {
        "exp":     exp_id,
        "time":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "git":     git,
        "gpu":     _gpu_name(),
        "python":  sys.version.split()[0],
        "torch":   _torch_version(),
        "params":  params,
        "metrics": metrics,
    }

    # Per-experiment JSON
    json_path = os.path.join(results_dir, f"{exp_id}_{git}.json")
    with open(json_path, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"[resultlog] → {json_path}")

    # Manifest
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest_path = os.path.join(repo_root, "results", "manifest.jsonl")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")

    # RESULTS.md
    results_md = os.path.join(repo_root, "results", "RESULTS.md")
    with open(results_md, "a") as f:
        f.write(f"\n### {exp_id} ({rec['time']}, {rec['gpu']}, git={git})\n")
        f.write("```json\n")
        f.write(json.dumps(metrics, indent=2, default=str))
        f.write("\n```\n")

    return json_path
