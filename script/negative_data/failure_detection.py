from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FailureDetectionResult:
    actor_keys: list[str]
    common_length: int
    positive_mean: np.ndarray
    positive_std: np.ndarray
    positive_std_norm: np.ndarray
    actor_z_scores: np.ndarray
    actor_thresholds: np.ndarray
    threshold_ratios: np.ndarray
    score: np.ndarray
    max_actor_indices: np.ndarray
    onset_frame: int | None


def first_persistent_crossing(values, threshold, persistence_frames):
    run_length = 0
    for frame, value in enumerate(values):
        if np.isfinite(value) and value > threshold:
            run_length += 1
            if run_length >= persistence_frames:
                return frame - persistence_frames + 1
        else:
            run_length = 0
    return None


def detect_failure_onset(
    positive_positions,
    failed_positions,
    actor_keys,
    std_floor_m=0.005,
    threshold_k=3.0,
    warmup_frames=3,
    initial_margin=1.1,
    persistence_frames=3,
):
    if len(positive_positions) == 0:
        raise ValueError("At least one positive trace is required")
    if std_floor_m <= 0:
        raise ValueError("std_floor_m must be positive")
    if threshold_k <= 0:
        raise ValueError("threshold_k must be positive")
    if warmup_frames < 1:
        raise ValueError("warmup_frames must be positive")
    if initial_margin < 1:
        raise ValueError("initial_margin must be at least 1")
    if persistence_frames < 1:
        raise ValueError("persistence_frames must be positive")

    actor_count = len(actor_keys)
    traces = [np.asarray(trace, dtype=np.float64) for trace in positive_positions]
    failed = np.asarray(failed_positions, dtype=np.float64)
    for trace in traces + [failed]:
        if trace.ndim != 3 or trace.shape[1:] != (actor_count, 3):
            raise ValueError(
                "Position traces must have shape [frames, actors, 3]"
            )

    common_length = min([len(trace) for trace in traces] + [len(failed)])
    if common_length == 0:
        raise ValueError("Position traces must contain at least one frame")
    positive_stack = np.stack(
        [trace[:common_length] for trace in traces], axis=0
    )
    failed = failed[:common_length]

    positive_mean = np.nanmean(positive_stack, axis=0)
    positive_std = np.nanstd(positive_stack, axis=0)
    positive_std_norm = np.linalg.norm(positive_std, axis=2)
    deviation = np.linalg.norm(failed - positive_mean, axis=2)
    actor_z_scores = deviation / np.maximum(
        positive_std_norm, std_floor_m
    )

    warmup_length = min(warmup_frames, common_length)
    initial_z = actor_z_scores[:warmup_length]
    initial_baseline = np.nanmax(initial_z, axis=0)
    initial_baseline = np.nan_to_num(
        initial_baseline, nan=0.0, posinf=0.0, neginf=0.0
    )
    actor_thresholds = np.maximum(
        threshold_k, initial_margin * initial_baseline
    )
    threshold_ratios = actor_z_scores / actor_thresholds[None, :]
    finite_ratios = np.where(
        np.isfinite(threshold_ratios), threshold_ratios, -np.inf
    )
    max_actor_indices = np.argmax(finite_ratios, axis=1)
    score = np.max(finite_ratios, axis=1)
    onset_frame = first_persistent_crossing(
        score, threshold=1.0, persistence_frames=persistence_frames
    )

    return FailureDetectionResult(
        actor_keys=list(actor_keys),
        common_length=common_length,
        positive_mean=positive_mean,
        positive_std=positive_std,
        positive_std_norm=positive_std_norm,
        actor_z_scores=actor_z_scores,
        actor_thresholds=actor_thresholds,
        threshold_ratios=threshold_ratios,
        score=score,
        max_actor_indices=max_actor_indices,
        onset_frame=onset_frame,
    )
