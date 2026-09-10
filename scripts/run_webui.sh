#!/usr/bin/env bash
# Launch the Fire3D Gradio WebUI: one uploaded RGB image -> textured 3D scene.
#
# Requires the environment built by scripts/install.sh, with the WebUI extras:
#   FIRE3D_WITH_WEBUI=1 bash scripts/install.sh
#
# Every flag of `fire3d serve` is forwarded, e.g.:
#   bash scripts/run_webui.sh --port 7860 --auth user:secret
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${FIRE3D_VENV_DIR:-$ROOT/.venv}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "[webui] $VENV_DIR is missing; run 'bash scripts/install.sh' first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec python -m fire3d.webui.app "$@"
