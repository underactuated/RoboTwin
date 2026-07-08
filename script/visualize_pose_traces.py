import sys

sys.path.append("./")

from argparse import ArgumentParser
from pathlib import Path

import numpy as np


POSE_COMPONENTS = {
    "x": 0,
    "y": 1,
    "z": 2,
    "qw": 3,
    "qx": 4,
    "qy": 5,
    "qz": 6,
}


def load_traces(trace_dir, run_kinds):
    trace_dir = Path(trace_dir)
    files = sorted(trace_dir.glob("*.npz"))
    traces = []
    for path in files:
        if not any(path.name.startswith(f"{kind}_") for kind in run_kinds):
            continue
        with np.load(path) as data:
            traces.append(
                {
                    "path": path,
                    "poses": data["poses"],
                    "actor_keys": data["actor_keys"].astype(str).tolist(),
                    "success": bool(data["final_success"]),
                    "amplitude": float(data["actual_amplitude"]),
                    "attempt": int(data["replay_attempt"]),
                    "run_kind": "recorded"
                    if path.name.startswith("recorded_")
                    else "dry",
                }
            )
    if not traces:
        kinds = ", ".join(run_kinds)
        raise ValueError(f"No {kinds} NPZ traces found in {trace_dir}")
    return traces


def select_actors(traces, requested):
    available = []
    for trace in traces:
        for actor in trace["actor_keys"]:
            if actor not in available:
                available.append(actor)
    if requested is None:
        return available
    missing = [actor for actor in requested if actor not in available]
    if missing:
        raise ValueError(
            f"Unknown actors: {missing}. Available actors: {available}"
        )
    return requested


def actor_component(trace, actor, component_index):
    try:
        actor_index = trace["actor_keys"].index(actor)
    except ValueError:
        return None
    return trace["poses"][:, actor_index, component_index]


def trace_label(trace):
    outcome = "success" if trace["success"] else "failure"
    return (
        f"{trace['run_kind']} #{trace['attempt']} "
        f"amp={trace['amplitude']:g} {outcome}"
    )


def plot_individual(ax, traces, actor, component_index):
    for trace in traces:
        values = actor_component(trace, actor, component_index)
        if values is None:
            continue
        color = "tab:green" if trace["success"] else "tab:red"
        linestyle = "--" if trace["run_kind"] == "recorded" else "-"
        ax.plot(
            np.arange(len(values)),
            values,
            color=color,
            linestyle=linestyle,
            alpha=0.75,
            linewidth=1.2,
            label=trace_label(trace),
        )


def plot_aggregated_positives(ax, traces, actor, component_index):
    positives = []
    for trace in traces:
        if not trace["success"]:
            continue
        values = actor_component(trace, actor, component_index)
        if values is not None:
            positives.append(values)

    if positives:
        common_length = min(len(values) for values in positives)
        values = np.stack([values[:common_length] for values in positives])
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        frames = np.arange(common_length)
        ax.plot(
            frames,
            mean,
            color="tab:blue",
            linewidth=1.5,
            label=f"positive mean (n={len(positives)})",
        )
        ax.fill_between(
            frames,
            mean - std,
            mean + std,
            color="tab:blue",
            alpha=0.2,
            label="positive ±1 std",
        )

    for trace in traces:
        if trace["success"]:
            continue
        values = actor_component(trace, actor, component_index)
        if values is None:
            continue
        linestyle = "--" if trace["run_kind"] == "recorded" else "-"
        ax.plot(
            np.arange(len(values)),
            values,
            color="tab:red",
            linestyle=linestyle,
            alpha=0.8,
            linewidth=1.2,
            label=trace_label(trace),
        )


def make_figure(
    traces,
    actors,
    components,
    output_path,
    aggregate_positives=False,
    dpi=150,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required: install it with `pip install matplotlib`"
        ) from exc

    figure, axes = plt.subplots(
        len(actors),
        len(components),
        figsize=(3.4 * len(components), 2.4 * len(actors)),
        sharex=True,
        squeeze=False,
    )

    for row, actor in enumerate(actors):
        for column, component in enumerate(components):
            ax = axes[row, column]
            component_index = POSE_COMPONENTS[component]
            if aggregate_positives:
                plot_aggregated_positives(
                    ax, traces, actor, component_index
                )
            else:
                plot_individual(ax, traces, actor, component_index)
            if row == 0:
                ax.set_title(component)
            if column == 0:
                ax.set_ylabel(actor)
            if row == len(actors) - 1:
                ax.set_xlabel("frame")
            ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        figure.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            ncol=min(4, len(unique)),
            fontsize="small",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_parser():
    parser = ArgumentParser(
        description="Visualize actor pose components from replay trace NPZ files."
    )
    parser.add_argument(
        "trace_dir",
        help="Episode trace directory containing dry_*.npz files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path (default: <trace_dir>/pose_traces.png).",
    )
    parser.add_argument(
        "--actors",
        nargs="+",
        default=None,
        help="Actor keys to plot (default: all).",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=tuple(POSE_COMPONENTS),
        default=list(POSE_COMPONENTS),
        help="Pose components to plot (default: all).",
    )
    parser.add_argument(
        "--aggregate-positives",
        action="store_true",
        help="Plot successful mean ± std and failed traces individually.",
    )
    parser.add_argument(
        "--run-kinds",
        nargs="+",
        choices=("dry", "recorded"),
        default=("dry",),
        help="Trace kinds to include (default: dry).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main():
    args = build_parser().parse_args()
    traces = load_traces(args.trace_dir, args.run_kinds)
    actors = select_actors(traces, args.actors)
    output = args.output or str(Path(args.trace_dir) / "pose_traces.png")
    output = make_figure(
        traces,
        actors,
        args.components,
        output,
        aggregate_positives=args.aggregate_positives,
        dpi=args.dpi,
    )
    print(f"Saved pose visualization: {output}")
    print(f"Traces: {len(traces)}, actors: {len(actors)}, components: {len(args.components)}")
    if args.aggregate_positives and any(
        component.startswith("q") for component in args.components
    ):
        print(
            "Note: quaternion components are aggregated numerically; "
            "orientation-angle statistics require sign-aware quaternion handling."
        )


if __name__ == "__main__":
    main()
