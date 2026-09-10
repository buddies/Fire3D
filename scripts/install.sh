#!/usr/bin/env bash
# Build the Fire3D environment with pyenv plus a project-local virtualenv.
#
# pyenv supplies only the interpreter. The CUDA 12.8 toolkit (with `nvcc`), a
# C++ compiler, and git are host requirements and are checked below; cmake and
# ninja are installed into the venv when the host has no usable copy.
#
#   bash scripts/install.sh                       # inference + training
#   FIRE3D_WITH_WEBUI=1 bash scripts/install.sh   # also the Gradio WebUI
#   FIRE3D_PYTHON_VERSION=3.11.15 bash scripts/install.sh
#   FIRE3D_ATTENTION_BACKEND=xformers bash scripts/install.sh   # no CUDA build
#   FIRE3D_BUILD_JOBS=1 bash scripts/install.sh   # tiny memory limit
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The released protocols were validated on Python 3.10 and 3.11 is supported as
# well, so the interpreter pyenv already resolves -- its global, or this
# checkout's .python-version -- is reused when it is 3.10/3.11; one is only
# installed when the machine has neither. Set FIRE3D_PYTHON_VERSION to pin an
# exact build.
PYTHON_SERIES="${FIRE3D_PYTHON_SERIES:-3.11}"
SUPPORTED_PYTHON='^3\.(10|11)(\.[0-9]+)?$'
ATTENTION_BACKEND="${FIRE3D_ATTENTION_BACKEND:-flash_attn}"
FLASH_ATTN_VERSION="${FIRE3D_FLASH_ATTN_VERSION:-2.7.3}"
VENV_DIR="${FIRE3D_VENV_DIR:-$ROOT/.venv}"
MIN_CMAKE="3.28"
DINOV3_COMMIT="31703e4cbf1ccb7c4a72daa1350405f86754b6d1"

# Activation hooks and shell profiles append to these variables without
# guarding against `set -u`, so default them before anything activates or calls
# into the toolchain.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

version_gte() {  # version_gte HAVE REQUIRED
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]
}

if ! command -v git >/dev/null 2>&1; then
  echo "[install] git is required: DINOv3 and Eigen are checked out from source." >&2
  exit 1
fi
if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
  echo "[install] a C++ compiler is required to build the CUDA extensions." >&2
  exit 1
fi

if ! command -v pyenv >/dev/null 2>&1; then
  cat >&2 <<'MESSAGE'
[install] pyenv is required. Install it, then restart your shell:
  curl -fsSL https://pyenv.run | bash
MESSAGE
  exit 1
fi
eval "$(pyenv init -)"

installed_line() {  # newest interpreter of a series that pyenv already has
  pyenv versions --bare | tr -d ' ' | grep -E "^$1\.[0-9]+$" | sort -V | tail -n 1
}

available_line() {  # newest interpreter of a series this pyenv can install
  pyenv install --list | tr -d ' ' | grep -E "^$1\.[0-9]+$" | sort -V | tail -n 1
}

other_series() {
  if [[ "$PYTHON_SERIES" == "3.11" ]]; then echo 3.10; else echo 3.11; fi
}

resolve_python_version() {
  local candidate
  if [[ -n "${FIRE3D_PYTHON_VERSION:-}" ]]; then
    echo "$FIRE3D_PYTHON_VERSION"
    return
  fi
  # What pyenv already selects here (global or .python-version), then any
  # installed interpreter of a supported series.
  for candidate in \
    "$(pyenv version-name 2>/dev/null | head -n 1)" \
    "$(pyenv global 2>/dev/null | head -n 1)" \
    "$(installed_line "$PYTHON_SERIES")" \
    "$(installed_line "$(other_series)")"; do
    if [[ "$candidate" =~ $SUPPORTED_PYTHON ]]; then
      echo "$candidate"
      return
    fi
  done
  # Nothing usable on the machine: install the newest of the preferred series.
  available_line "$PYTHON_SERIES" || available_line "$(other_series)"
}

PYTHON_VERSION="$(resolve_python_version)"
if ! [[ "$PYTHON_VERSION" =~ $SUPPORTED_PYTHON ]]; then
  echo "[install] unusable Python '$PYTHON_VERSION': install 3.10.x or 3.11.x with pyenv," >&2
  echo "[install] or set FIRE3D_PYTHON_VERSION to a supported build." >&2
  exit 1
fi
# A patched request fixes which series the "version unavailable" fallback uses.
if [[ "$PYTHON_VERSION" =~ ^([0-9]+\.[0-9]+)\. ]]; then
  PYTHON_SERIES="${BASH_REMATCH[1]}"
fi

if ! pyenv versions --bare | tr -d ' ' | grep -qx "$PYTHON_VERSION"; then
  echo "[install] pyenv install $PYTHON_VERSION"
  if ! pyenv install -s "$PYTHON_VERSION"; then
    # The requested patch release can be newer than this pyenv's definitions.
    fallback="$(available_line "$PYTHON_SERIES" || available_line "$(other_series)")"
    if [[ -z "$fallback" ]]; then
      echo "[install] pyenv has no ${PYTHON_SERIES}.x definition; run 'pyenv update'." >&2
      exit 1
    fi
    echo "[install] $PYTHON_VERSION is unavailable; using $fallback instead." >&2
    PYTHON_VERSION="$fallback"
    pyenv install -s "$PYTHON_VERSION"
  fi
fi
PYTHON_BIN="$(pyenv prefix "$PYTHON_VERSION")/bin/python"

# The venv pins the interpreter for the environment; `.python-version` pins it
# for pyenv itself, so a bare `python` inside the checkout matches the build the
# venv was created from.
echo "$PYTHON_VERSION" > "$ROOT/.python-version"

VENV_PYTHON_VERSION=""
if [[ -x "$VENV_DIR/bin/python" ]]; then
  VENV_PYTHON_VERSION="$(
    "$VENV_DIR/bin/python" -c 'import sys; print(".".join(str(p) for p in sys.version_info[:3]))'
  )"
fi
if [[ -n "$VENV_PYTHON_VERSION" && "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION" ]]; then
  echo "[install] $VENV_DIR holds Python $VENV_PYTHON_VERSION, not $PYTHON_VERSION; recreating it." >&2
  rm -rf "$VENV_DIR"
  VENV_PYTHON_VERSION=""
fi
if [[ -z "$VENV_PYTHON_VERSION" ]]; then
  echo "[install] creating $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# pyenv cannot supply the CUDA toolkit, so resolve nvcc -- and CUDA_HOME for the
# extension builds -- before spending time on downloads that need it.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if command -v nvcc >/dev/null 2>&1; then
  NVCC="$(command -v nvcc)"
  # Follow a distribution symlink so CUDA_HOME is the real toolkit prefix.
  NVCC_REAL="$(readlink -f "$NVCC" 2>/dev/null || echo "$NVCC")"
  CUDA_HOME="$(dirname "$(dirname "$NVCC_REAL")")"
elif [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
  NVCC="$CUDA_HOME/bin/nvcc"
else
  cat >&2 <<MESSAGE
[install] nvcc was not found. Install the CUDA 12.8 toolkit, make sure
[install] \$CUDA_HOME/bin is on PATH (CUDA_HOME=$CUDA_HOME), then re-run.
MESSAGE
  exit 1
fi
export CUDA_HOME
echo "[install] nvcc: $("$NVCC" --version | tail -n 1) (CUDA_HOME=$CUDA_HOME)"
export PATH="$(dirname "$NVCC"):$PATH"

if ! command -v cmake >/dev/null 2>&1; then
  python -m pip install "cmake>=${MIN_CMAKE}"
elif ! version_gte "$(cmake --version | head -n 1 | awk '{print $3}')" "$MIN_CMAKE"; then
  echo "[install] cmake is older than ${MIN_CMAKE}; installing a newer one into the venv." >&2
  python -m pip install --upgrade "cmake>=${MIN_CMAKE}"
fi
if ! command -v ninja >/dev/null 2>&1; then
  python -m pip install ninja
fi

# Every source build below compiles CUDA kernels with ninja, which defaults to
# one nvcc per core at several GB each. In a memory-limited container that is
# what gets the process OOM-killed (exit 137), so cap the parallelism instead of
# letting the build size itself to the core count.
export MAX_JOBS="${FIRE3D_BUILD_JOBS:-2}"
export NVCC_THREADS="${FIRE3D_NVCC_THREADS:-1}"
echo "[install] build parallelism: MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS"

python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "$ROOT[dev]"
if [[ "${FIRE3D_WITH_WEBUI:-0}" == "1" ]]; then
  python -m pip install -e "$ROOT[webui]"
fi

python -m pip install spconv-cu118==2.3.8 --no-build-isolation

# Attention backend. `pip install flash-attn` has no wheel on PyPI, so it
# compiles the whole CUDA kernel suite -- by far the most memory-hungry step in
# this script. The project publishes prebuilt wheels per torch/ABI/Python
# combination, so try that first and only fall back to a bounded source build.
install_flash_attn_wheel() {
  local wheel_url
  if [[ -n "${FIRE3D_FLASH_ATTN_WHEEL_URL:-}" ]]; then
    echo "[install] using FIRE3D_FLASH_ATTN_WHEEL_URL: $FIRE3D_FLASH_ATTN_WHEEL_URL"
    python -m pip install "$FIRE3D_FLASH_ATTN_WHEEL_URL"
    return
  fi
  # Resolve the whole prebuilt-wheel name in one probe, and fail loudly: this
  # function is called from an `if` condition, where bash suspends `errexit`, so
  # a failed probe would otherwise yield an empty ABI tag and a bogus URL --
  # silently turning a 2-minute wheel install into a multi-hour source build.
  wheel="$(
    FLASH_ATTN_VERSION="$FLASH_ATTN_VERSION" python - <<'PROBE'
import os
import sys

import torch

version = os.environ["FLASH_ATTN_VERSION"]
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
series = ".".join(torch.__version__.split(".")[:2])
tag = "cp{}{}".format(*sys.version_info[:2])
print(f"flash_attn-{version}+cu12torch{series}cxx11abi{abi}-{tag}-{tag}-linux_x86_64.whl")
PROBE
  )" || {
    echo "[install] could not determine the flash-attn wheel name from torch." >&2
    return 1
  }
  echo "[install] trying the prebuilt flash-attn wheel: $wheel"
  python -m pip install \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${wheel}"
}

case "$ATTENTION_BACKEND" in
  flash_attn | flash_attn_3)
    if ! install_flash_attn_wheel; then
      cat >&2 <<MESSAGE
[install] no prebuilt flash-attn wheel for this torch/ABI/Python combination at
[install] v${FLASH_ATTN_VERSION} (Fire3D pins torch 2.7.1, and the published
[install] wheel set only covers the combinations upstream built). Building from
[install] source now -- that needs several GB of RAM and 1-3 hours of CPU, and is
[install] what gets a memory-limited container OOM-killed (exit 137). Prefer:
[install]   FIRE3D_ATTENTION_BACKEND=xformers bash scripts/install.sh   # no CUDA build
[install]   FIRE3D_BUILD_JOBS=1 bash scripts/install.sh                # serial build
[install] or point FIRE3D_FLASH_ATTN_WHEEL_URL at a wheel you host yourself.
MESSAGE
      if ! python -m pip install --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"; then
        cat >&2 <<MESSAGE
[install] flash-attn could not be installed. Exit code 137 means the container
[install] was killed for exceeding its memory limit; raise the limit, lower
[install] FIRE3D_BUILD_JOBS further, or re-run with the xformers backend.
MESSAGE
        exit 1
      fi
    fi
    ;;
  xformers)
    # xformers ships prebuilt wheels, so this path compiles nothing. Fire3D
    # reads the backend from the environment at import time, so the run has to
    # export it as well -- printed below.
    python -m pip install xformers
    echo "[install] installed the xformers attention backend."
    echo "[install] export these in the serving environment so both attention paths use it:"
    echo "[install]   export ATTN_BACKEND=xformers"
    echo "[install]   export SPARSE_ATTN_BACKEND=xformers"
    ;;
  *)
    echo "[install] unknown FIRE3D_ATTENTION_BACKEND='$ATTENTION_BACKEND' (use flash_attn, flash_attn_3, or xformers)." >&2
    exit 1
    ;;
esac
python -m pip install --no-build-isolation \
  'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae' \
  'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47' \
  'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8' \
  'git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e'

DINO_DIR="$ROOT/third_party/dinov3"
if [[ ! -d "$DINO_DIR/.git" ]]; then
  git clone https://github.com/facebookresearch/dinov3.git "$DINO_DIR"
fi
git -C "$DINO_DIR" fetch origin "$DINOV3_COMMIT"
git -C "$DINO_DIR" checkout --detach "$DINOV3_COMMIT"

for eigen_dir in \
  "$ROOT/trellis2_x2/CuMesh/third_party/cubvh/third_party/eigen" \
  "$ROOT/trellis2_x2/o-voxel/third_party/eigen"; do
  if [[ ! -d "$eigen_dir/Eigen" ]]; then
    rm -rf "$eigen_dir"
    git clone --depth 1 --branch 3.4.0 \
      https://gitlab.com/libeigen/eigen.git "$eigen_dir"
  fi
done

python -m pip install --no-build-isolation --no-deps "$ROOT/trellis2_x2/CuMesh"
python -m pip install --no-build-isolation --no-deps "$ROOT/trellis2_x2/o-voxel"

REL_VENV="${VENV_DIR#"$ROOT"/}"
echo "Fire3D environment ready (Python $PYTHON_VERSION). Activate it with:"
echo "  source $REL_VENV/bin/activate"
