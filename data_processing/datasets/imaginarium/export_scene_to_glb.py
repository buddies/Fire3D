import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BLENDER_PATH = os.environ.get(
    "FIRE3D_BLENDER",
    str(REPO_ROOT / "blender/blender-4.5.1-linux-x64/blender"),
)

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        raise FileNotFoundError(
            f"Blender is missing at {BLENDER_PATH}; run scripts/install_blender.sh "
            "or set FIRE3D_BLENDER"
        )


def _to_glb(file_path, output_dir):
    to_glb_script_path = os.path.join(os.path.dirname(__file__), 'to_glb.py')
    file_path = os.path.expanduser(file_path)

    # Note: for `.blend` we must pass it to `-b` so Blender opens it before running `-P`.
    args = [BLENDER_PATH, '-b']
    if file_path.endswith('.blend'):
        args.append(file_path)

    args += [
        '-P', to_glb_script_path,
        '--',
        '--object', file_path,
        '--output_folder', output_dir,
    ]

    call(args)
    # call(args, stdout=DEVNULL, stderr=DEVNULL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True,
                        help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The path the output will be dumped to.')
    parser.add_argument('--blender-path', type=Path, default=Path(BLENDER_PATH))
    opt = parser.parse_args()
    opt = edict(vars(opt))
    BLENDER_PATH = str(opt.blender_path.expanduser().resolve())

    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    os.makedirs(opt.output_dir, exist_ok=True)

    _to_glb(opt.file_path, opt.output_dir)
