from contextlib import contextmanager
from pathlib import Path
import shutil

import numpy as np

from .actors import discover_task_actors


class ActorTraceRecorder:
    """Record task-actor poses as [x, y, z, qw, qx, qy, qz]."""

    def __init__(self, task, include_nonshiftable_actors=False):
        actors = discover_task_actors(task)
        self.actors = (
            actors
            if include_nonshiftable_actors
            else [actor for actor in actors if actor.shiftable]
        )
        self.frames = []

    def append(self):
        frame = np.full((len(self.actors), 7), np.nan, dtype=np.float64)
        for index, actor in enumerate(self.actors):
            try:
                pose = actor.entity.get_pose()
                frame[index, :3] = np.asarray(pose.p, dtype=np.float64)
                frame[index, 3:] = np.asarray(pose.q, dtype=np.float64)
            except Exception:
                continue
        self.frames.append(frame)

    def to_array(self):
        if not self.frames:
            return np.empty((0, len(self.actors), 7), dtype=np.float64)
        return np.stack(self.frames)


@contextmanager
def record_at_dataset_frames(task, recorder):
    """Append a trace frame whenever the dataset capture hook is invoked."""
    original_take_picture = task._take_picture

    def take_picture_with_trace():
        recorder.append()
        return original_take_picture()

    task._take_picture = take_picture_with_trace
    try:
        yield
    finally:
        task._take_picture = original_take_picture


class ReplayTraceWriter:
    """Replay observer that writes raw pose traces for exploratory inspection."""

    def __init__(
        self,
        output_dir,
        include_nonshiftable_actors=False,
        episode_complete_callback=None,
    ):
        self.output_dir = Path(output_dir)
        self.include_nonshiftable_actors = include_nonshiftable_actors
        self.episode_complete_callback = episode_complete_callback

    def start(self, task, context):
        return ActorTraceRecorder(
            task,
            include_nonshiftable_actors=self.include_nonshiftable_actors,
        )

    def reset_episode(self, episode_idx):
        episode_dir = self.output_dir / f"episode_{episode_idx:03d}"
        if episode_dir.exists():
            shutil.rmtree(episode_dir)

    def finish(self, recorder, context, final_success):
        episode_dir = self.output_dir / f"episode_{context['episode_idx']:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        amplitude = context["actual_amplitude"]
        run_kind = "recorded" if context["save_data"] else "dry"
        outcome = "success" if final_success else "failure"
        filename = (
            f"{run_kind}_attempt_{context['replay_attempt']:03d}_"
            f"amp{amplitude:g}_{outcome}.npz"
        )
        np.savez_compressed(
            episode_dir / filename,
            poses=recorder.to_array(),
            actor_keys=np.asarray([actor.key for actor in recorder.actors]),
            entity_names=np.asarray(
                [actor.entity_name or "" for actor in recorder.actors]
            ),
            shiftable=np.asarray(
                [actor.shiftable for actor in recorder.actors], dtype=np.bool_
            ),
            episode_idx=np.asarray(context["episode_idx"]),
            seed=np.asarray(context["seed"]),
            trajectory_idx=np.asarray(context["traj_idx"]),
            actual_amplitude=np.asarray(amplitude),
            replay_attempt=np.asarray(context["replay_attempt"]),
            save_data=np.asarray(context["save_data"]),
            final_success=np.asarray(final_success),
        )

    def on_episode_complete(self, episode_idx):
        if self.episode_complete_callback is not None:
            self.episode_complete_callback(
                self.output_dir / f"episode_{episode_idx:03d}"
            )
