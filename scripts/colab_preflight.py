"""
Colab setup preflight.

Checks whether the compiled numeric stack can actually import and use NumPy's
random extension. If the stack is broken, repairs NumPy/Pandas/SciPy/PyArrow/
Scikit-learn together and exits with code 10 to request a runtime restart.
"""
import subprocess
import sys


NUMERIC_CHECK = (
    "import numpy as np; np.random.seed(0); "
    "import pandas, pyarrow, scipy, sklearn, torch; "
    "print('numpy', np.__version__); "
    "print('pandas', pandas.__version__); "
    "print('pyarrow', pyarrow.__version__); "
    "print('scipy', scipy.__version__); "
    "print('sklearn', sklearn.__version__); "
    "print('torch', torch.__version__)"
)

REPAIR_PACKAGES = [
    "numpy>=2.0,<2.3",
    "pandas>=2.2",
    "pyarrow>=16",
    "scipy>=1.13",
    "scikit-learn>=1.5",
]


def run(args, check=False):
    proc = subprocess.run(args, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args, proc.stdout, proc.stderr)
    return proc


def check_numeric_stack():
    return run([sys.executable, "-c", NUMERIC_CHECK], check=False)


def repair_numeric_stack():
    return run(
        [
            sys.executable, "-m", "pip", "install",
            "--force-reinstall", "--no-cache-dir", "--progress-bar", "off",
            *REPAIR_PACKAGES,
        ],
        check=True,
    )


def main():
    print("[preflight] Checking numeric binary stack...")
    first = check_numeric_stack()
    if first.returncode == 0:
        print("[preflight] Numeric stack OK.")
        return 0

    print("[preflight] Numeric stack is broken; repairing compiled wheels.")
    repair_numeric_stack()

    print("[preflight] Rechecking after repair...")
    second = check_numeric_stack()
    if second.returncode != 0:
        print("[preflight] Repair did not validate cleanly.", file=sys.stderr)
        return second.returncode

    print("=" * 78)
    print("[preflight] Numeric stack repaired.")
    print("[preflight] IMPORTANT: restart the Colab runtime now, then rerun E0 setup.")
    print("[preflight] Do not continue to E0 schema audit in this same runtime.")
    print("=" * 78)
    return 10


if __name__ == "__main__":
    raise SystemExit(main())
