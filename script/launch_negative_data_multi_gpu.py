#!/usr/bin/env python3
"""Launch negative RoboTwin data collection jobs across multiple GPUs.

This is intentionally a thin scheduler around script/collect_negative_data.py.
It does not collect data itself. For each task it creates a per-job task_config
YAML that overrides only the output root and episode count, then launches the
existing collector in a subprocess with CUDA_VISIBLE_DEVICES set to one GPU.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Editable defaults.
#
# These values are used when the corresponding CLI flags are not provided.
# Edit this block for routine generation runs instead of typing many flags.
# ---------------------------------------------------------------------------

DEFAULT_TASKS = [
    "hanging_mug",
]
DEFAULT_TASK_CONFIG = "my_config"
DEFAULT_EPISODES_PER_TASK = 2 #30
DEFAULT_OUTPUT_ROOT = "./data_temp" #"./data"
DEFAULT_MAX_GPUS = 2
DEFAULT_GPU_IDS = None  # Example: [0, 1, 2, 3]. None means range(max_gpus).
DEFAULT_SEED_START = 0
DEFAULT_SEED_STRIDE_PER_TASK = 0

DEFAULT_NEGATIVE_MODE = "replay_object_shift"
DEFAULT_PERTURBATION_AMPLITUDE = 0.25
DEFAULT_REPLAY_SHIFT_GROWTH_FACTOR = 2.0
DEFAULT_REPLAY_SHIFT_MAX_AMPLITUDE = 8.0

# Available fields: task, config, mode, amplitude, run_id.
DEFAULT_SAVE_SETTING_TEMPLATE = "{config}_negative_{mode}_amp{amplitude}"

DEFAULT_LOG_DIR = "logs/negative_collection"
DEFAULT_POLL_SECONDS = 10.0
DEFAULT_KEEP_JOB_CONFIGS = True
DEFAULT_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


COLLECTOR = Path("script/collect_negative_data.py")
TASK_CONFIG_DIR = Path("task_config")
UPDATE_PATH_SCRIPT = Path("script/.update_path.sh")


@dataclass
class Job:
    task: str
    config_name: str
    save_setting: str
    seed_start: int
    output_root: Path
    log_path: Path
    config_written: bool = False


@dataclass
class RunningJob:
    job: Job
    gpu_id: int
    process: subprocess.Popen
    log_file: object


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)
    return value.strip("_")


def format_float(value: float) -> str:
    return f"{value:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Schedule negative RoboTwin sample generation across multiple GPUs."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task subset to run. Defaults to DEFAULT_TASKS in this file.",
    )
    parser.add_argument(
        "--task-config",
        default=DEFAULT_TASK_CONFIG,
        help="Base task config YAML name without .yml.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=DEFAULT_EPISODES_PER_TASK,
        help="Number of accepted negative episodes requested per task.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory. Final path is <root>/<task>/<save_setting>.",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=DEFAULT_MAX_GPUS,
        help="Maximum number of GPUs to use.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="Explicit GPU ids. Overrides DEFAULT_GPU_IDS/range(max_gpus).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED_START,
        help="First simulator seed to try for the first task.",
    )
    parser.add_argument(
        "--seed-stride-per-task",
        type=int,
        default=DEFAULT_SEED_STRIDE_PER_TASK,
        help=(
            "Added to seed-start for each subsequent task. Use a value larger "
            "than expected tries per task to avoid overlapping seed ranges."
        ),
    )
    parser.add_argument(
        "--negative-mode",
        choices=["planner_perturb", "replay_object_shift"],
        default=DEFAULT_NEGATIVE_MODE,
    )
    parser.add_argument(
        "--perturbation-amplitude",
        type=float,
        default=DEFAULT_PERTURBATION_AMPLITUDE,
    )
    parser.add_argument(
        "--replay-shift-growth-factor",
        type=float,
        default=DEFAULT_REPLAY_SHIFT_GROWTH_FACTOR,
    )
    parser.add_argument(
        "--replay-shift-max-amplitude",
        type=float,
        default=DEFAULT_REPLAY_SHIFT_MAX_AMPLITUDE,
    )
    parser.add_argument(
        "--save-setting-template",
        default=DEFAULT_SAVE_SETTING_TEMPLATE,
        help=(
            "Format string for output setting. Fields: task, config, mode, "
            "amplitude, run_id."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for subprocess stdout/stderr logs.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Identifier used in generated config names and templates.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help="Scheduler polling interval.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated configs and commands without launching jobs.",
    )
    parser.add_argument(
        "--cleanup-job-configs",
        action="store_true",
        default=not DEFAULT_KEEP_JOB_CONFIGS,
        help="Remove generated per-job task_config YAMLs after successful jobs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    if args.max_gpus <= 0:
        raise ValueError("--max-gpus must be positive")
    if args.perturbation_amplitude < 0:
        raise ValueError("--perturbation-amplitude must be non-negative")
    if args.seed_start < 0:
        raise ValueError("--seed-start must be non-negative")
    if args.seed_stride_per_task < 0:
        raise ValueError("--seed-stride-per-task must be non-negative")
    if args.negative_mode == "replay_object_shift":
        if args.perturbation_amplitude <= 0:
            raise ValueError(
                "--perturbation-amplitude must be positive for replay_object_shift"
            )
        if args.replay_shift_growth_factor <= 1:
            raise ValueError("--replay-shift-growth-factor must be greater than 1")
        if args.replay_shift_max_amplitude < args.perturbation_amplitude:
            raise ValueError(
                "--replay-shift-max-amplitude must be at least perturbation amplitude"
            )
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")


def resolve_gpus(args: argparse.Namespace) -> list[int]:
    if args.gpus is not None:
        gpus = args.gpus
    elif DEFAULT_GPU_IDS is not None:
        gpus = DEFAULT_GPU_IDS
    else:
        gpus = list(range(args.max_gpus))
    if not gpus:
        raise ValueError("At least one GPU id is required")
    return gpus[: args.max_gpus]


def load_base_config(config_name: str) -> dict:
    path = TASK_CONFIG_DIR / f"{config_name}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Base task config not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_job_config_name(
    *,
    base_config_name: str,
    task: str,
    run_id: str,
) -> str:
    return sanitize_name(
        f"_negative_batch_{run_id}_{base_config_name}_{task}"
    )


def write_job_config(
    *,
    config_name: str,
    base_config: dict,
    output_root: Path,
    episodes_per_task: int,
) -> None:
    config = dict(base_config)
    config["save_path"] = str(output_root)
    config["episode_num"] = int(episodes_per_task)

    config_path = TASK_CONFIG_DIR / f"{config_name}.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def make_save_setting(args: argparse.Namespace, task: str) -> str:
    return sanitize_name(
        args.save_setting_template.format(
            task=task,
            config=args.task_config,
            mode=args.negative_mode,
            amplitude=format_float(args.perturbation_amplitude),
            run_id=args.run_id,
        )
    )


def make_command(args: argparse.Namespace, job: Job) -> list[str]:
    command = [
        sys.executable,
        str(COLLECTOR),
        job.task,
        job.config_name,
        "--negative-mode",
        args.negative_mode,
        "--perturbation-amplitude",
        format_float(args.perturbation_amplitude),
        "--save-setting",
        job.save_setting,
        "--seed-start",
        str(job.seed_start),
    ]
    if args.negative_mode == "replay_object_shift":
        command.extend(
            [
                "--replay-shift-growth-factor",
                format_float(args.replay_shift_growth_factor),
                "--replay-shift-max-amplitude",
                format_float(args.replay_shift_max_amplitude),
            ]
        )
    return command


def maybe_update_paths() -> None:
    if UPDATE_PATH_SCRIPT.exists():
        subprocess.run(
            ["bash", str(UPDATE_PATH_SCRIPT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def build_jobs(args: argparse.Namespace, *, write_configs: bool) -> list[Job]:
    tasks = args.tasks or DEFAULT_TASKS
    if not tasks:
        raise ValueError("No tasks configured")

    output_root = Path(args.output_root).expanduser()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    TASK_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_base_config(args.task_config)
    jobs = []
    for task_index, task in enumerate(tasks):
        config_name = make_job_config_name(
            base_config_name=args.task_config,
            task=task,
            run_id=args.run_id,
        )
        if write_configs:
            write_job_config(
                config_name=config_name,
                base_config=base_config,
                output_root=output_root,
                episodes_per_task=args.episodes_per_task,
            )
        save_setting = make_save_setting(args, task)
        log_path = log_dir / (
            f"{sanitize_name(args.run_id)}_"
            f"{sanitize_name(task)}_"
            f"{sanitize_name(save_setting)}.log"
        )
        jobs.append(
            Job(
                task=task,
                config_name=config_name,
                save_setting=save_setting,
                seed_start=args.seed_start + task_index * args.seed_stride_per_task,
                output_root=output_root,
                log_path=log_path,
                config_written=write_configs,
            )
        )
    return jobs


def launch_job(args: argparse.Namespace, job: Job, gpu_id: int) -> RunningJob:
    command = make_command(args, job)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = open(job.log_path, "w", encoding="utf-8")
    log_file.write(f"Command: {' '.join(command)}\n")
    log_file.write(f"CUDA_VISIBLE_DEVICES={gpu_id}\n")
    log_file.flush()

    process = subprocess.Popen(
        command,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(
        f"Started task={job.task} gpu={gpu_id} "
        f"save={job.output_root / job.task / job.save_setting} "
        f"log={job.log_path}"
    )
    return RunningJob(job=job, gpu_id=gpu_id, process=process, log_file=log_file)


def cleanup_job_config(job: Job) -> None:
    if not job.config_written:
        return
    config_path = TASK_CONFIG_DIR / f"{job.config_name}.yml"
    try:
        config_path.unlink()
    except FileNotFoundError:
        pass


def run_scheduler(args: argparse.Namespace, jobs: list[Job], gpus: list[int]) -> int:
    pending = list(jobs)
    running: dict[int, RunningJob] = {}
    failures: list[tuple[Job, int]] = []

    while pending or running:
        for gpu_id in gpus:
            if gpu_id not in running and pending:
                running[gpu_id] = launch_job(args, pending.pop(0), gpu_id)

        finished = []
        for gpu_id, state in running.items():
            return_code = state.process.poll()
            if return_code is None:
                continue

            state.log_file.close()
            if return_code == 0:
                print(f"Finished task={state.job.task} gpu={gpu_id}")
                if args.cleanup_job_configs:
                    cleanup_job_config(state.job)
            else:
                failures.append((state.job, return_code))
                print(
                    f"FAILED task={state.job.task} gpu={gpu_id} "
                    f"return_code={return_code} log={state.job.log_path}"
                )
            finished.append(gpu_id)

        for gpu_id in finished:
            del running[gpu_id]

        if pending or running:
            time.sleep(args.poll_seconds)

    if failures:
        print("\nFailed jobs:")
        for job, return_code in failures:
            print(f"  task={job.task} return_code={return_code} log={job.log_path}")
        return 1

    print("All jobs finished successfully.")
    return 0


def print_plan(args: argparse.Namespace, jobs: list[Job], gpus: list[int]) -> None:
    print("Negative data multi-GPU launch plan")
    print(f"Run id: {args.run_id}")
    print(f"GPUs: {gpus}")
    print(f"Episodes per task: {args.episodes_per_task}")
    print(f"Output root: {args.output_root}")
    print(f"Negative mode: {args.negative_mode}")
    print(f"Perturbation amplitude: {format_float(args.perturbation_amplitude)}")
    for job in jobs:
        print("")
        print(f"Task: {job.task}")
        print(f"Seed start: {job.seed_start}")
        print(f"Generated config: {TASK_CONFIG_DIR / (job.config_name + '.yml')}")
        print(f"Output: {job.output_root / job.task / job.save_setting}")
        print(f"Log: {job.log_path}")
        print("Command:")
        print("  " + " ".join(make_command(args, job)))


def main() -> int:
    args = parse_args()
    validate_args(args)
    gpus = resolve_gpus(args)
    maybe_update_paths()
    jobs = build_jobs(args, write_configs=not args.dry_run)
    print_plan(args, jobs, gpus)
    if args.dry_run:
        return 0
    return run_scheduler(args, jobs, gpus)


if __name__ == "__main__":
    raise SystemExit(main())
