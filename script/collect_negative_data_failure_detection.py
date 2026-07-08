import sys

sys.path.append("./")

import json
from argparse import ArgumentParser
from pathlib import Path

from script.analyze_replay_failure_detection import analyze_trace_directory
from script.create_failure_detection_video import create_video
from script.collect_negative_data import (
    REPLAY_OBJECT_SHIFT,
    class_decorator,
    collect_negative_data,
    prepare_args,
)
from script.negative_data import ReplayTraceWriter


def main(
    task_name,
    task_config,
    episode_num=1,
    perturbation_amplitude=0.25,
    replay_shift_growth_factor=2.0,
    replay_shift_max_amplitude=8.0,
    save_setting=None,
    trace_debug_dir=None,
    include_nonshiftable_actors=False,
    target_positive_traces=5,
    min_positive_traces=3,
    positive_attempt_budget=20,
    backoff_factor=0.5,
    min_probe_amplitude_ratio=0.01,
    std_floor_m=0.005,
    threshold_k=3.0,
    warmup_frames=3,
    initial_margin=1.1,
    persistence_frames=3,
    create_detection_video=False,
    detection_video_width=960,
):
    if episode_num <= 0:
        raise ValueError("episode-num must be positive")
    if perturbation_amplitude <= 0:
        raise ValueError("perturbation-amplitude must be positive")
    if replay_shift_growth_factor <= 1:
        raise ValueError("replay-shift-growth-factor must be greater than 1")
    if replay_shift_max_amplitude < perturbation_amplitude:
        raise ValueError(
            "replay-shift-max-amplitude must be at least perturbation-amplitude"
        )
    if target_positive_traces < 1:
        raise ValueError("target-positive-traces must be positive")
    if not 1 <= min_positive_traces <= target_positive_traces:
        raise ValueError(
            "min-positive-traces must be between 1 and target-positive-traces"
        )
    if positive_attempt_budget < 0:
        raise ValueError("positive-attempt-budget must be non-negative")
    if not 0 < backoff_factor < 1:
        raise ValueError("backoff-factor must be between 0 and 1")
    if not 0 < min_probe_amplitude_ratio < 1:
        raise ValueError("min-probe-amplitude-ratio must be between 0 and 1")
    if std_floor_m <= 0:
        raise ValueError("std-floor-m must be positive")
    if threshold_k <= 0:
        raise ValueError("threshold-k must be positive")
    if warmup_frames < 1:
        raise ValueError("warmup-frames must be positive")
    if initial_margin < 1:
        raise ValueError("initial-margin must be at least 1")
    if persistence_frames < 1:
        raise ValueError("persistence-frames must be positive")
    if detection_video_width < 320:
        raise ValueError("detection-video-width must be at least 320")

    detector_parameters = {
        "std_floor_m": std_floor_m,
        "threshold_k": threshold_k,
        "warmup_frames": warmup_frames,
        "initial_margin": initial_margin,
        "persistence_frames": persistence_frames,
    }

    if save_setting is None:
        save_setting = (
            f"{task_config}_negative_failure_detection_"
            f"amp{perturbation_amplitude:g}"
        )

    args = prepare_args(
        task_name,
        task_config,
        perturbation_amplitude,
        REPLAY_OBJECT_SHIFT,
        replay_shift_growth_factor=replay_shift_growth_factor,
        replay_shift_max_amplitude=replay_shift_max_amplitude,
        save_setting=save_setting,
    )
    args["episode_num"] = episode_num
    args["failure_detector_target_positive_traces"] = target_positive_traces
    args["failure_detector_min_positive_traces"] = min_positive_traces
    args["failure_detector_positive_attempt_budget"] = positive_attempt_budget
    args["failure_detector_backoff_factor"] = backoff_factor
    args[
        "failure_detector_min_probe_amplitude_ratio"
    ] = min_probe_amplitude_ratio
    if trace_debug_dir is None:
        trace_debug_dir = f"{args['save_path']}/failure_detection_debug"

    print("======= Failure-Detection Collection Config =======")
    print(f"Task: {task_name}")
    print(f"Config: {task_config}")
    print(f"Episodes: {episode_num}")
    print(f"Perturbation amplitude: {perturbation_amplitude:g}")
    print(f"Replay shift growth factor: {replay_shift_growth_factor:g}")
    print(f"Replay shift max amplitude: {replay_shift_max_amplitude:g}")
    print(f"Save path: {args['save_path']}")
    print(f"Trace debug path: {trace_debug_dir}")
    print(f"Target positive traces: {target_positive_traces}")
    print(f"Minimum positive traces: {min_positive_traces}")
    print(f"Positive attempt budget: {positive_attempt_budget}")
    print(f"Detector parameters: {detector_parameters}")
    print("Source plans: generated lazily")
    print("Failure detection: enabled")
    print("===================================================")

    task = class_decorator(task_name)
    def analyze_episode(trace_episode_dir):
        try:
            summary = analyze_trace_directory(
                trace_episode_dir,
                **detector_parameters,
            )
        except ValueError as error:
            summary = {
                **detector_parameters,
                "detected_failure_onset_frame": None,
                "max_actor_at_onset": None,
                "confidence": "unavailable",
                "error": str(error),
            }
            with open(
                trace_episode_dir / "failure_detection.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(summary, file, indent=2)
        print(
            "automatic failure detection: "
            f"episode = {trace_episode_dir.name}, "
            f"onset = {summary['detected_failure_onset_frame']}, "
            f"actor = {summary['max_actor_at_onset']}, "
            f"confidence = {summary['confidence']}"
        )
        if create_detection_video and summary["confidence"] != "unavailable":
            episode_idx = int(trace_episode_dir.name.rsplit("_", 1)[1])
            output_path = trace_episode_dir / "failure_detection_video.mp4"
            create_video(
                Path(args["save_path"]) / "video" / f"episode{episode_idx}.mp4",
                trace_episode_dir / "failure_detection_stats.npz",
                output_path,
                output_width=detection_video_width,
            )
            print(f"saved combined detection video: {output_path}")

    collect_negative_data(
        task,
        args,
        replay_observer=ReplayTraceWriter(
            trace_debug_dir,
            include_nonshiftable_actors=include_nonshiftable_actors,
            episode_complete_callback=analyze_episode,
        ),
    )


def build_parser():
    parser = ArgumentParser(
        description=(
            "Experimental replay-object-shift collector for automatic "
            "failure-onset detection."
        )
    )
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument(
        "--episode-num",
        type=int,
        default=1,
        help="Number of failed episodes to collect (default: 1).",
    )
    parser.add_argument(
        "--perturbation-amplitude",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--replay-shift-growth-factor",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--replay-shift-max-amplitude",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--save-setting",
        default=None,
        help=(
            "Output directory name under data/<task>. Defaults to a distinct "
            "experimental directory."
        ),
    )
    parser.add_argument(
        "--trace-debug-dir",
        default=None,
        help="Raw NPZ trace directory (default: inside the output directory).",
    )
    parser.add_argument(
        "--include-nonshiftable-actors",
        action="store_true",
        help="Include task actors that are not eligible for object shifting.",
    )
    parser.add_argument(
        "--target-positive-traces",
        type=int,
        default=5,
        help="Successful shifted traces required per failure (default: 5).",
    )
    parser.add_argument(
        "--min-positive-traces",
        type=int,
        default=3,
        help="Reject failures with fewer successful envelope traces.",
    )
    parser.add_argument(
        "--positive-attempt-budget",
        type=int,
        default=20,
        help="Maximum post-failure envelope probes (default: 20).",
    )
    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=0.5,
        help="Amplitude multiplier when no successful lower bound exists.",
    )
    parser.add_argument(
        "--min-probe-amplitude-ratio",
        type=float,
        default=0.01,
        help="Stop backoff below this fraction of the base amplitude.",
    )
    parser.add_argument("--std-floor-m", type=float, default=0.005)
    parser.add_argument("--threshold-k", type=float, default=3.0)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--initial-margin", type=float, default=1.1)
    parser.add_argument("--persistence-frames", type=int, default=3)
    parser.add_argument(
        "--create-detection-video",
        action="store_true",
        help="Create a replay video with a moving failure-score strip.",
    )
    parser.add_argument(
        "--detection-video-width",
        type=int,
        default=960,
        help="Combined detection video width (default: 960).",
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()

    from test_render import Sapien_TEST

    Sapien_TEST()

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main(
        task_name=parsed.task_name,
        task_config=parsed.task_config,
        episode_num=parsed.episode_num,
        perturbation_amplitude=parsed.perturbation_amplitude,
        replay_shift_growth_factor=parsed.replay_shift_growth_factor,
        replay_shift_max_amplitude=parsed.replay_shift_max_amplitude,
        save_setting=parsed.save_setting,
        trace_debug_dir=parsed.trace_debug_dir,
        include_nonshiftable_actors=parsed.include_nonshiftable_actors,
        target_positive_traces=parsed.target_positive_traces,
        min_positive_traces=parsed.min_positive_traces,
        positive_attempt_budget=parsed.positive_attempt_budget,
        backoff_factor=parsed.backoff_factor,
        min_probe_amplitude_ratio=parsed.min_probe_amplitude_ratio,
        std_floor_m=parsed.std_floor_m,
        threshold_k=parsed.threshold_k,
        warmup_frames=parsed.warmup_frames,
        initial_margin=parsed.initial_margin,
        persistence_frames=parsed.persistence_frames,
        create_detection_video=parsed.create_detection_video,
        detection_video_width=parsed.detection_video_width,
    )
