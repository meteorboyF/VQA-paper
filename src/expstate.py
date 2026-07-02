"""
Experiment completion state (skip-if-done for Run-All).

Each experiment writes a DONE.json marker into its Drive-backed results dir
when it finishes successfully. On the next run (including "Run all cells"),
the experiment checks the marker plus the artifacts it is supposed to have
produced, and skips instantly if everything is present.

Rules:
  - FORCE_RERUN (config / VQA_FORCE_RERUN=1) bypasses every marker.
  - A marker alone is NOT enough: the required artifact paths for the
    *current* config must also exist. If you add a backbone later, the
    experiment reruns even though DONE.json exists.
  - Deleting <results_dir>/DONE.json forces a single experiment to rerun.
"""
import json
import os
import time


def _marker_path(results_dir: str) -> str:
    return os.path.join(results_dir, "DONE.json")


def is_done(exp_id: str, results_dir: str, required=()) -> bool:
    """True if the experiment completed before and its artifacts still exist."""
    from src import config
    # Re-read the env var at call time: the notebook may set VQA_FORCE_RERUN
    # after src.config was first imported in this session.
    if config.FORCE_RERUN or os.environ.get("VQA_FORCE_RERUN", "0") == "1":
        print(f"[{exp_id}] FORCE_RERUN=True - ignoring DONE marker.")
        return False
    marker = _marker_path(results_dir)
    if not os.path.exists(marker):
        return False
    try:
        with open(marker) as f:
            info = json.load(f)
    except Exception:
        print(f"[{exp_id}] DONE marker unreadable - rerunning: {marker}")
        return False
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"[{exp_id}] DONE marker found but {len(missing)} required "
              f"artifact(s) are missing - rerunning.")
        for p in missing[:8]:
            print(f"  missing: {p}")
        return False
    return True


def mark_done(exp_id: str, results_dir: str, artifacts=(), extra=None) -> str:
    """Write the DONE marker after a successful run."""
    os.makedirs(results_dir, exist_ok=True)
    info = {
        "exp": exp_id,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "artifacts": [str(p) for p in artifacts],
    }
    if extra:
        info["extra"] = extra
    try:
        import subprocess
        info["git"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["git"] = "nogit"
    marker = _marker_path(results_dir)
    tmp = marker + ".tmp"
    with open(tmp, "w") as f:
        json.dump(info, f, indent=2)
    os.replace(tmp, marker)
    print(f"[{exp_id}] DONE marker written -> {marker}")
    return marker


def skip_banner(exp_id: str, results_dir: str) -> None:
    """Print why an experiment is being skipped and how to force a rerun."""
    when = "?"
    try:
        with open(_marker_path(results_dir)) as f:
            when = json.load(f).get("time", "?")
    except Exception:
        pass
    print("=" * 70)
    print(f"[{exp_id} SKIP] Already completed on {when}. Cached results on Drive:")
    print(f"  {results_dir}")
    print(f"  To redo: delete {os.path.join(results_dir, 'DONE.json')} "
          f"or set VQA_FORCE_RERUN=1 before running.")
    print("=" * 70)
