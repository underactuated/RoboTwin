import sys

sys.path.append("./")

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from script.negative_data.failure_detection import detect_failure_onset


def load_trace(path):
    with np.load(path) as data:
        return {
            "path": path,
            "poses": data["poses"],
            "actor_keys": data["actor_keys"].astype(str).tolist(),
            "success": bool(data["final_success"]),
            "amplitude": float(data["actual_amplitude"]),
            "attempt": int(data["replay_attempt"]),
            "seed": int(data["seed"]),
        }


def select_traces(trace_dir):
    trace_dir = Path(trace_dir)
    dry_traces = [
        load_trace(path) for path in sorted(trace_dir.glob("dry_*.npz"))
    ]
    positives = [trace for trace in dry_traces if trace["success"]]
    recorded_failures = [
        load_trace(path)
        for path in sorted(trace_dir.glob("recorded_*_failure.npz"))
    ]
    if recorded_failures:
        failed = recorded_failures[0]
    else:
        failures = [trace for trace in dry_traces if not trace["success"]]
        if not failures:
            raise ValueError(f"No failed trace found in {trace_dir}")
        failed = failures[0]
    if not positives:
        raise ValueError(f"No successful dry traces found in {trace_dir}")

    expected_keys = failed["actor_keys"]
    incompatible = [
        trace["path"].name
        for trace in positives
        if trace["actor_keys"] != expected_keys
    ]
    if incompatible:
        raise ValueError(
            "Actor keys/order differ in positive traces: "
            + ", ".join(incompatible)
        )
    return positives, failed


def save_outputs(
    output_dir,
    positives,
    failed,
    result,
    parameters,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onset = result.onset_frame
    onset_actor = (
        result.actor_keys[result.max_actor_indices[onset]]
        if onset is not None
        else None
    )
    summary = {
        "num_positive_traces": len(positives),
        "positive_trace_files": [
            trace["path"].name for trace in positives
        ],
        "positive_amplitudes": [
            trace["amplitude"] for trace in positives
        ],
        "failed_trace_file": failed["path"].name,
        "failed_amplitude": failed["amplitude"],
        "seed": failed["seed"],
        "actor_keys": result.actor_keys,
        "trace_lengths": {
            trace["path"].name: int(len(trace["poses"]))
            for trace in positives + [failed]
        },
        "common_length": result.common_length,
        **parameters,
        "actor_thresholds": {
            actor: float(threshold)
            for actor, threshold in zip(
                result.actor_keys, result.actor_thresholds
            )
        },
        "detected_failure_onset_frame": onset,
        "max_actor_at_onset": onset_actor,
        "score_at_onset": (
            float(result.score[onset]) if onset is not None else None
        ),
        "confidence": "normal" if len(positives) >= 3 else "low",
    }
    with open(
        output_dir / "failure_detection.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    np.savez_compressed(
        output_dir / "failure_detection_stats.npz",
        actor_keys=np.asarray(result.actor_keys),
        positive_mean=result.positive_mean,
        positive_std=result.positive_std,
        positive_std_norm=result.positive_std_norm,
        actor_z_scores=result.actor_z_scores,
        actor_thresholds=result.actor_thresholds,
        threshold_ratios=result.threshold_ratios,
        score=result.score,
        max_actor_indices=result.max_actor_indices,
        onset_frame=np.asarray(-1 if onset is None else onset),
    )
    return summary


def save_plot(output_path, result):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required: install it with `pip install matplotlib`"
        ) from exc

    figure, (score_ax, actor_ax) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True
    )
    frames = np.arange(result.common_length)
    score_ax.plot(frames, result.score, color="black", label="max z/threshold")
    score_ax.axhline(1.0, color="tab:red", linestyle="--", label="threshold")
    score_ax.set_ylabel("threshold ratio")
    score_ax.grid(alpha=0.25)
    score_ax.legend()

    for actor_index, actor in enumerate(result.actor_keys):
        actor_ax.plot(
            frames,
            result.actor_z_scores[:, actor_index],
            label=actor,
        )
        actor_ax.axhline(
            result.actor_thresholds[actor_index],
            linestyle="--",
            alpha=0.5,
        )
    actor_ax.set_xlabel("frame")
    actor_ax.set_ylabel("normalized deviation z")
    actor_ax.grid(alpha=0.25)
    actor_ax.legend()

    if result.onset_frame is not None:
        for ax in (score_ax, actor_ax):
            ax.axvline(
                result.onset_frame,
                color="tab:purple",
                linestyle=":",
                label="detected onset",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_parser():
    parser = ArgumentParser(
        description="Detect failure onset from saved replay pose traces."
    )
    parser.add_argument("trace_dir")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: trace directory).",
    )
    parser.add_argument("--std-floor-m", type=float, default=0.005)
    parser.add_argument("--threshold-k", type=float, default=3.0)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--initial-margin", type=float, default=1.1)
    parser.add_argument("--persistence-frames", type=int, default=3)
    return parser


def analyze_trace_directory(trace_dir, output_dir=None, **parameters):
    positives, failed = select_traces(trace_dir)
    result = detect_failure_onset(
        [trace["poses"][:, :, :3] for trace in positives],
        failed["poses"][:, :, :3],
        failed["actor_keys"],
        **parameters,
    )
    output_dir = Path(output_dir or trace_dir)
    summary = save_outputs(
        output_dir, positives, failed, result, parameters
    )
    save_plot(output_dir / "failure_detection_score.png", result)
    return summary


def main():
    args = build_parser().parse_args()
    parameters = {
        "std_floor_m": args.std_floor_m,
        "threshold_k": args.threshold_k,
        "warmup_frames": args.warmup_frames,
        "initial_margin": args.initial_margin,
        "persistence_frames": args.persistence_frames,
    }
    output_dir = Path(args.output_dir or args.trace_dir)
    summary = analyze_trace_directory(
        args.trace_dir,
        output_dir=output_dir,
        **parameters,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved failure-detection outputs to: {output_dir}")


if __name__ == "__main__":
    main()
