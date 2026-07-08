import numpy as np
import sapien.core as sapien

from .actors import discover_task_actors, is_shiftable_entity, unwrap_entity


class ReplayObjectShifter:
    def __init__(self, amplitude, seed, max_objects=1):
        self.amplitude = float(amplitude)
        self.rng = np.random.default_rng(seed)
        self.max_objects = max(1, int(max_objects))
        self.xy_scale = 0.03
        self.z_scale = 0.0
        self.events = []

    @property
    def enabled(self):
        return self.amplitude > 0

    def normal(self, scale, size=None):
        return self.rng.normal(0.0, self.amplitude * scale, size=size)

    def apply(self, task):
        if not self.enabled:
            return []
        candidates = self.find_candidates(task)
        if not candidates:
            return []
        self.rng.shuffle(candidates)
        shifted = []
        for path, _source, entity in candidates:
            if len(shifted) >= self.max_objects:
                break
            event = self.shift_entity(path, entity)
            if event is not None:
                shifted.append(event)
        settle_steps = int(getattr(task, "replay_shift_settle_steps", 20))
        for _ in range(settle_steps):
            task.scene.step()
        self.events = shifted
        return shifted

    def find_candidates(self, task):
        return [
            (actor.key, actor.source, actor.entity)
            for actor in discover_task_actors(task)
            if actor.shiftable
        ]

    unwrap_entity = staticmethod(unwrap_entity)
    is_shiftable_entity = staticmethod(is_shiftable_entity)

    def shift_entity(self, path, entity):
        pose = entity.get_pose()
        old_p = np.asarray(pose.p, dtype=np.float64)
        delta = np.array(
            [
                self.normal(self.xy_scale),
                self.normal(self.xy_scale),
                self.normal(self.z_scale),
            ],
            dtype=np.float64,
        )
        new_x = old_p[0] + delta[0]
        if old_p[0] != 0 and old_p[0] * new_x < 0:
            delta[0] = -0.5 * old_p[0]
        new_p = old_p + delta
        new_p[0] = np.clip(new_p[0], -0.42, 0.42)
        new_p[1] = np.clip(new_p[1], -0.32, 0.32)
        try:
            entity.set_pose(sapien.Pose(new_p, pose.q))
        except Exception:
            return None
        return {
            "name": "replay_object_shift",
            "value": {
                "path": path,
                "entity_name": entity.get_name(),
                "old_position": old_p.tolist(),
                "delta": delta.tolist(),
                "new_position": new_p.tolist(),
            },
        }
