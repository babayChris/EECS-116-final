#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VIO_PYTHON=${VIO_PYTHON:-python3}
"$VIO_PYTHON" - <<'PY'
import sys
if sys.version_info[:2] != (3, 14):
    raise SystemExit("This pinned environment uses Python 3.14. Set VIO_PYTHON to your Python 3.14 executable.")
PY
git submodule update --init --recursive
if [ ! -x .venv-vio/bin/python ]; then
  "$VIO_PYTHON" -m venv .venv-vio
fi
.venv-vio/bin/python - <<'PY'
import sys
if sys.version_info[:2] != (3, 14):
    raise SystemExit("The existing .venv-vio is not Python 3.14. Use a fresh checkout or recreate that environment.")
PY
mkdir -p work
if command -v uv >/dev/null 2>&1; then
  UV_CACHE_DIR="$PWD/work/uv-cache" uv pip install --python .venv-vio/bin/python -r requirements-host.txt
else
  PIP_CACHE_DIR="$PWD/work/pip-cache" .venv-vio/bin/python -m pip install -r requirements-host.txt
fi
.venv-vio/bin/viobench --help >/dev/null
echo "Ready. First run: bash run_local_vio.sh host klt_imu 1"
