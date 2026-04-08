"""
s2t_fs_v2 — Stateless experimental harness for the FASTT manuscript.

See reports/EXPERIMENTAL_PLAN.md for the locked design decisions.
This package is a sidecar to s2t_fs/; it imports model classes from there
but does not modify any existing code.

Entry points:
  python -m s2t_fs_v2.runner    # run one (dataset, method, n_train, seed) job
  python -m s2t_fs_v2.aggregate # collect result JSONs into a markdown table
"""

# ─────────────────────────────────────────────────────────────────────────────
# macOS OpenMP workaround
#
# On macOS, PyTorch ships with libomp and XGBoost ships with its own libomp.
# Loading both into the same process causes a hard segfault (SIGSEGV) the
# moment XGBoost runs after a torch op — exactly what FASTTAlternating does
# (torch transform → xgboost selector). Forcing single-thread OpenMP is the
# only stable workaround that doesn't require uninstalling either package.
#
# On Linux/Colab this is a no-op (libgomp on Linux does not clash with the
# libomp shipped in the manylinux torch wheel) and we want XGBoost to use
# all CPU cores there for speed, so we only set this on Darwin.
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
import platform as _platform

if _platform.system() == "Darwin":
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
