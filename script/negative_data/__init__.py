"""Shared utilities for negative-data collection."""

from .actors import TaskActor, discover_task_actors, unwrap_entity
from .replay_shift import ReplayObjectShifter
from .trace_recording import (
    ActorTraceRecorder,
    ReplayTraceWriter,
    record_at_dataset_frames,
)

__all__ = [
    "ActorTraceRecorder",
    "ReplayObjectShifter",
    "ReplayTraceWriter",
    "TaskActor",
    "discover_task_actors",
    "record_at_dataset_frames",
    "unwrap_entity",
]
