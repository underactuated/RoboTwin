import sys

sys.path.append("./")

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def draw_score_strip(
    width,
    height,
    scores,
    actor_keys,
    max_actor_indices,
    score_index,
    onset_frame,
):
    strip = np.full((height, width, 3), 248, dtype=np.uint8)
    left, right, top, bottom = 58, 18, 42, 32
    plot_width = width - left - right
    plot_height = height - top - bottom
    finite_scores = scores[np.isfinite(scores)]
    score_max = max(
        1.25,
        float(np.max(finite_scores)) * 1.08 if finite_scores.size else 1.25,
    )

    def frame_x(frame):
        denominator = max(1, len(scores) - 1)
        return left + int(round(frame / denominator * plot_width))

    def score_y(score):
        clipped = float(np.clip(score, 0.0, score_max))
        return top + plot_height - int(round(clipped / score_max * plot_height))

    if onset_frame >= 0:
        onset_x = frame_x(onset_frame)
        cv2.rectangle(
            strip,
            (onset_x, top),
            (left + plot_width, top + plot_height),
            (245, 235, 255),
            thickness=-1,
        )

    cv2.line(
        strip,
        (left, score_y(1.0)),
        (left + plot_width, score_y(1.0)),
        (40, 40, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        strip,
        "threshold",
        (left + 5, score_y(1.0) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (40, 40, 220),
        1,
        cv2.LINE_AA,
    )

    points = []
    for frame, score in enumerate(scores):
        if np.isfinite(score):
            points.append((frame_x(frame), score_y(score)))
    if len(points) >= 2:
        cv2.polylines(
            strip,
            [np.asarray(points, dtype=np.int32)],
            isClosed=False,
            color=(25, 25, 25),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    if onset_frame >= 0:
        onset_x = frame_x(onset_frame)
        cv2.line(
            strip,
            (onset_x, top),
            (onset_x, top + plot_height),
            (180, 40, 150),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            strip,
            f"onset {onset_frame}",
            (min(onset_x + 6, width - 120), top + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 40, 150),
            1,
            cv2.LINE_AA,
        )

    cursor_x = frame_x(score_index)
    cv2.line(
        strip,
        (cursor_x, top),
        (cursor_x, top + plot_height),
        (220, 130, 20),
        3,
        cv2.LINE_AA,
    )

    current_score = scores[score_index]
    actor_index = int(max_actor_indices[score_index])
    actor = actor_keys[actor_index] if actor_keys else "unknown"
    status = "FAILURE" if onset_frame >= 0 and score_index >= onset_frame else "pre-failure"
    status_color = (30, 30, 220) if status == "FAILURE" else (50, 120, 50)
    cv2.putText(
        strip,
        (
            f"frame {score_index}/{len(scores) - 1}   "
            f"ratio {current_score:.2f}   actor {actor}"
        ),
        (left, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        strip,
        status,
        (width - 135, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        strip,
        (left, top),
        (left + plot_width, top + plot_height),
        (80, 80, 80),
        1,
    )
    return strip


def create_video(
    video_path,
    stats_path,
    output_path,
    output_width=960,
    strip_height=220,
):
    with np.load(stats_path) as stats:
        scores = stats["score"].astype(np.float64)
        actor_keys = stats["actor_keys"].astype(str).tolist()
        max_actor_indices = stats["max_actor_indices"]
        onset_frame = int(stats["onset_frame"])

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")
    input_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0 or input_width <= 0 or input_height <= 0 or fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid input video metadata: {video_path}")

    video_height = int(round(input_height * output_width / input_width))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, video_height + strip_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video: {output_path}")

    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        score_index = int(
            round(written * (len(scores) - 1) / max(1, frame_count - 1))
        )
        frame = cv2.resize(
            frame,
            (output_width, video_height),
            interpolation=cv2.INTER_LINEAR,
        )
        strip = draw_score_strip(
            output_width,
            strip_height,
            scores,
            actor_keys,
            max_actor_indices,
            score_index,
            onset_frame,
        )
        writer.write(np.vstack((frame, strip)))
        written += 1

    capture.release()
    writer.release()
    if written == 0:
        raise RuntimeError("No video frames were decoded")
    return written, fps


def build_parser():
    parser = ArgumentParser(
        description="Combine a replay video with its failure-detection score."
    )
    parser.add_argument("dataset_dir")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--video", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--strip-height", type=int, default=220)
    return parser


def main():
    args = build_parser().parse_args()
    if args.width < 320:
        raise ValueError("width must be at least 320 pixels")
    if args.strip_height < 140:
        raise ValueError("strip-height must be at least 140 pixels")

    dataset_dir = Path(args.dataset_dir)
    trace_dir = (
        dataset_dir
        / "failure_detection_debug"
        / f"episode_{args.episode:03d}"
    )
    video_path = Path(
        args.video or dataset_dir / "video" / f"episode{args.episode}.mp4"
    )
    stats_path = Path(
        args.stats or trace_dir / "failure_detection_stats.npz"
    )
    output_path = Path(
        args.output or trace_dir / "failure_detection_video.mp4"
    )
    written, fps = create_video(
        video_path,
        stats_path,
        output_path,
        output_width=args.width,
        strip_height=args.strip_height,
    )
    print(
        f"Saved combined failure-detection video: {output_path} "
        f"({written} frames, {fps:g} FPS)"
    )


if __name__ == "__main__":
    main()
