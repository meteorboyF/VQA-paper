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
    # A 0-byte artifact means a write was interrupted (Drive flush hazard):
    # treat it exactly like a missing file so the experiment reruns.
    missing = [p for p in required
               if not os.path.exists(p) or os.path.getsize(p) == 0]
    if missing:
        print(f"[{exp_id}] DONE marker found but {len(missing)} required "
              f"artifact(s) are missing or empty - rerunning.")
        for p in missing[:8]:
            print(f"  missing/empty: {p}")
        return False
    return True


def write_json_atomic(path: str, obj) -> None:
    """Write JSON via tmp-file + fsync + atomic replace.

    Plain json.dump(open(path, 'w')) on a Drive-backed path can leave a
    0-byte or truncated file if the runtime dies or Drive flushes lazily;
    downstream readers then crash with JSONDecodeError. Atomic replace
    guarantees the final path is either absent or complete.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json_valid(path: str):
    """Return parsed JSON, or None if the file is missing/empty/corrupt."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


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
