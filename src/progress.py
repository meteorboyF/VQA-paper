"""
Colab-friendly progress bars and failure diagnostics for notebook cells.

The helpers here are intentionally lightweight: they print enough context to
debug bad Drive paths, missing caches, GPU mismatches, and schema surprises
without hiding the original traceback.
"""
from contextlib import contextmanager
import os
import platform
import traceback


def _tqdm():
    try:
        from tqdm.auto import tqdm
    except Exception:
        from tqdm import tqdm
    return tqdm


@contextmanager
def cell(name: str, total: int):
    """Progress bar for one notebook cell; prints debug context on failure."""
    tqdm = _tqdm()
    bar = tqdm(total=total, desc=name, unit="step", leave=True)
    try:
        yield bar
        if bar.n < total:
            bar.update(total - bar.n)
        bar.set_description(f"{name} done")
    except Exception as exc:
        bar.set_description(f"{name} failed")
        bar.close()
        print_failure_context(name, exc)
        raise
    finally:
        bar.close()


def step(bar, message: str, advance: int = 1) -> None:
    """Advance a cell progress bar and show the current milestone."""
    try:
        bar.set_postfix_str(message[:80])
        bar.write(f"[progress] {message}")
        bar.update(advance)
    except Exception:
        print(f"[progress] {message}")


def print_failure_context(cell_name: str, exc: Exception) -> None:
    """Print actionable runtime context below a failed Colab cell."""
    print("\n" + "=" * 78)
    print(f"[ERROR CONTEXT] {cell_name}")
    print(f"Exception: {type(exc).__name__}: {exc}")
    print(f"CWD: {os.getcwd()}")
    print(f"Python: {platform.python_version()}  Platform: {platform.platform()}")
    try:
        import torch
        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU/no CUDA"
        print(f"Torch: {torch.__version__}  Device: {gpu}")
    except Exception as torch_exc:
        print(f"Torch unavailable/error: {torch_exc}")
    print("\nTraceback:")
    traceback.print_exc()
    print("\nUseful checks:")
    print("- If this is FileNotFoundError, compare the printed path with config.RAW_ZIPS and staged_dirs in results/E0_audit/audit.json.")
    print("- If this is a shape/index error, verify E1 split_ids.json and that E2/E4 caches were regenerated after code changes.")
    print("- If this is CUDA OOM, lower the batch size in the cell header and re-run; long cells resume from shards.")
    print("=" * 78 + "\n")


def install_error_hook(cell_name: str = "notebook cell") -> None:
    """Install an IPython exception hook that prints the same debug context."""
    try:
        ip = get_ipython()  # type: ignore[name-defined]
    except Exception:
        ip = None
    if ip is None:
        return

    def _handler(shell, etype, evalue, tb, tb_offset=None):
        print_failure_context(cell_name, evalue)
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

    try:
        ip.set_custom_exc((Exception,), _handler)
    except Exception:
        pass


def notebook_bar(name: str, total: int):
    """Create a tqdm.auto progress bar for a notebook cell."""
    return _tqdm()(total=total, desc=name, unit="step", leave=True)


def require_paths(label: str, paths) -> None:
    """Raise with a clear list of missing required files/dirs."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"[missing:{label}] Required paths were not found:")
        for p in missing:
            print(f"  - {p}")
        raise FileNotFoundError(f"{label}: {len(missing)} missing required path(s)")


def report_candidates(label: str, candidates) -> None:
    """Print candidate path existence, useful for Drive/unzip layout issues."""
    print(f"[candidates:{label}]")
    for p in candidates:
        mark = "OK" if os.path.exists(p) else "missing"
        print(f"  [{mark}] {p}")


def dataframe_summary(name: str, df) -> None:
    """Compact DataFrame summary for notebook output."""
    try:
        print(f"[df:{name}] rows={len(df)} cols={list(df.columns)}")
        if "split" in df.columns:
            print(f"[df:{name}] split counts={df['split'].value_counts().to_dict()}")
    except Exception as exc:
        print(f"[df:{name}] summary failed: {exc}")
