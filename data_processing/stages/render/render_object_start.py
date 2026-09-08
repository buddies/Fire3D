import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_object import (
    RENDER_ROOT,
    load_dataset_module,
    object_id_from_path,
)


RENDER_SCRIPT = Path(__file__).resolve().parent / "render_object.py"
from data_processing.objects import dataset_choices

DATASET_CHOICES = dataset_choices()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launcher: chunk un-rendered objects into groups and process them "
                    "with a ProcessPoolExecutor over a fixed pool of GPU-pinned slots.",
    )
    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug_random_10", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--workers_per_gpu", type=int, default=8)
    parser.add_argument("--objects_per_group", type=int, default=8)
    parser.add_argument(
        "--gpus", type=str, default="",
        help="Comma-separated GPU ids to use (overrides --num_gpus), e.g. '0,1,2,3'.",
    )
    parser.add_argument(
        "--cpu_threads_per_worker", type=int, default=1,
        help="Value used for OMP/MKL/OPENBLAS thread caps inside each worker.",
    )
    parser.add_argument(
        "--log_dir", type=str, default="",
        help="If set, redirect each group's stdout/stderr to <log_dir>/group_<G>.log.",
    )
    parser.add_argument(
        "--paths_dir", type=str, default="",
        help="Directory to write per-group paths JSON files. Defaults to a tempdir.",
    )
    return parser.parse_args()


def build_gpu_list(args):
    if args.gpus.strip():
        return [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    return list(range(args.num_gpus))


def filter_pending(model_paths, output_dataset_name):
    root = RENDER_ROOT / output_dataset_name
    pending = []
    for p in model_paths:
        cam = root / object_id_from_path(p) / "cameras.json"
        if not cam.exists():
            pending.append(p)
    return pending


def count_done(output_dataset_name):
    root = RENDER_ROOT / output_dataset_name
    if not root.exists():
        return 0
    return sum(1 for _ in root.glob("*/cameras.json"))


def chunk(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run_group(group_idx, paths_file, gpu_queue, common_args, cpu_threads, log_dir):
    gpu_id = gpu_queue.get()
    try:
        cmd = [
            sys.executable, "-u", str(RENDER_SCRIPT),
            "--paths_file", str(paths_file),
            "--gpu_id", str(gpu_id),
            *common_args,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["EGL_DEVICE_ID"] = "0"
        env["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM", "egl")
        t = str(cpu_threads)
        env["OMP_NUM_THREADS"] = t
        env["MKL_NUM_THREADS"] = t
        env["OPENBLAS_NUM_THREADS"] = t
        env["NUMEXPR_NUM_THREADS"] = t

        start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{start_ts}] launching group={group_idx} gpu={gpu_id} "
            f"cmd: {' '.join(cmd)}",
            flush=True,
        )

        if log_dir:
            log_path = Path(log_dir) / f"group_{group_idx:05d}.log"
            with open(log_path, "w") as fh:
                rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        else:
            rc = subprocess.run(cmd, env=env).returncode
        return group_idx, gpu_id, rc
    finally:
        gpu_queue.put(gpu_id)


def main():
    args = parse_args()
    gpu_list = build_gpu_list(args)
    if not gpu_list:
        raise SystemExit("No GPUs configured (set --num_gpus or --gpus).")
    pool_size = len(gpu_list) * args.workers_per_gpu

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    module_name, dataset = load_dataset_module(args.dataset_name)
    output_dataset_name = module_name

    model_paths = [Path(p) for p in dataset.list_all_model_paths()]
    if args.debug_random_10:
        import numpy as np
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(model_paths), size=min(10, len(model_paths)), replace=False)
        model_paths = [model_paths[i] for i in indices]

    total = len(model_paths)
    pending = filter_pending(model_paths, output_dataset_name)
    print(
        f"dataset={args.dataset_name} total={total} pending={len(pending)} "
        f"(skipping {total - len(pending)} with cameras.json present)",
        flush=True,
    )
    if not pending:
        print("Nothing to do.", flush=True)
        return

    if args.paths_dir:
        paths_root = Path(args.paths_dir)
        paths_root.mkdir(parents=True, exist_ok=True)
        cleanup_tmp = None
    else:
        tmp = tempfile.mkdtemp(prefix=f"render_groups_{output_dataset_name}_")
        paths_root = Path(tmp)
        cleanup_tmp = tmp

    groups = list(chunk(pending, args.objects_per_group))
    print(
        f"Splitting into {len(groups)} group(s) of up to {args.objects_per_group}; "
        f"pool_size={pool_size} over GPUs {gpu_list} ({args.workers_per_gpu}/GPU)",
        flush=True,
    )

    group_files = []
    for gi, group in enumerate(groups):
        f = paths_root / f"group_{gi:05d}.json"
        with open(f, "w") as fh:
            json.dump([str(p) for p in group], fh)
        group_files.append(f)

    common_args = [
        "--dataset_name", args.dataset_name,
        "--resolution", str(args.resolution),
        "--num_frames", str(args.num_frames),
        "--seed", str(args.seed),
    ]
    if args.overwrite:
        common_args.append("--overwrite")

    manager = Manager()
    gpu_queue = manager.Queue()
    for gpu_id in gpu_list:
        for _ in range(args.workers_per_gpu):
            gpu_queue.put(gpu_id)

    before = count_done(output_dataset_name)
    print(f"Already-rendered objects (cameras.json present): {before}", flush=True)

    failures = []
    start = time.time()
    completed = 0
    try:
        with ProcessPoolExecutor(max_workers=pool_size) as ex:
            futures = {
                ex.submit(
                    run_group, gi, group_files[gi], gpu_queue,
                    common_args, args.cpu_threads_per_worker,
                    str(log_dir) if log_dir else "",
                ): gi
                for gi in range(len(groups))
            }
            for fut in as_completed(futures):
                gi = futures[fut]
                try:
                    group_idx, gpu_id, rc = fut.result()
                except Exception as e:
                    failures.append((gi, -1, f"exception: {e}"))
                    print(f"[group {gi}] raised: {e}", flush=True)
                    continue
                completed += 1
                status = "ok" if rc == 0 else f"rc={rc}"
                if rc != 0:
                    failures.append((group_idx, gpu_id, rc))
                print(
                    f"[group {group_idx} gpu {gpu_id}] {status} "
                    f"({completed}/{len(groups)})",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("KeyboardInterrupt: shutting down pool...", flush=True)
        raise
    finally:
        if cleanup_tmp is not None:
            import shutil
            shutil.rmtree(cleanup_tmp, ignore_errors=True)

    after = count_done(output_dataset_name)
    elapsed = time.time() - start
    print(
        f"Done in {elapsed:.1f}s. cameras.json count: {before} -> {after} "
        f"(+{after - before}). Failed groups: {len(failures)}/{len(groups)}",
        flush=True,
    )
    if failures:
        for group_idx, gpu_id, rc in failures:
            print(f"  group={group_idx} gpu={gpu_id} rc={rc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
