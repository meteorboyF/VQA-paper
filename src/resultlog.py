"""
Experiment result logging.

Every experiment MUST call log_run() before finishing.
If log_run() wasn't called, the experiment didn't happen (per PIPELINE.md Sec 5).

Writes:
  results/EX_*/  <exp_id>_<git_hash>.json   - full metrics + params
  results/manifest.jsonl                    - one line per run (provenance)
  results/RESULTS.md                        - human-readable rolling summary
"""
import json
import os
import sys
import time
import subprocess


def _results_root_from(results_dir: str, repo_root: str = None) -> str:
    """
    Resolve the root results directory.

    In Colab, results_dir is usually Drive-backed, e.g.
    .../reliable_vqa_outputs/results/E3_triage. In that case manifest.jsonl
    and RESULTS.md must live beside those Drive results, not in ephemeral
    /content/VQA-paper/results.
    """
    abs_dir = os.path.abspath(results_dir)
    parent = os.path.dirname(abs_dir)
    if os.path.basename(parent) == "results":
        return parent
    if os.path.basename(abs_dir) == "results":
        return abs_dir
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "results")


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
    print(f"[resultlog] -> {json_path}")

    # Manifest beside the active results root (Drive-backed in Colab).
    results_root = _results_root_from(results_dir, repo_root=repo_root)
    manifest_path = os.path.join(results_root, "manifest.jsonl")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")

    # RESULTS.md
    results_md = os.path.join(results_root, "RESULTS.md")
    if not os.path.exists(results_md):
        with open(results_md, "w") as f:
            f.write("# RESULTS\n\nAuto-appended by resultlog.log_run() after each experiment.\n")
    with open(results_md, "a") as f:
        f.write(f"\n### {exp_id} ({rec['time']}, {rec['gpu']}, git={git})\n")
        f.write("```json\n")
        f.write(json.dumps(metrics, indent=2, default=str))
        f.write("\n```\n")

    return json_path
