#!/usr/bin/env bash
# Launch the Fire3D Gradio WebUI: one uploaded RGB image -> textured 3D scene.
#
# Requires the environment built by scripts/install.sh plus the WebUI extras:
#   python -m pip install -e ".[webui]"
#
# Every flag of `fire3d serve` is forwarded, e.g.:
#   bash scripts/run_webui.sh --port 7860 --auth user:secret
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${FIRE3D_ENV_NAME:-fire3d}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda activate "$ENV_NAME"
  fi
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec python -m fire3d.webui.app "$@"
