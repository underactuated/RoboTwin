from dataclasses import dataclass
from typing import Any

import numpy as np


_SKIP_TASK_ROOTS = {
    "engine", "renderer", "scene", "viewer", "robot", "cameras", "camera",
    "info", "now_obs", "world_pcd", "cluttered_objs", "record_cluttered_objects",
    "direction_light_lst", "point_light_lst", "file_path", "prohibited_area",
}


@dataclass(frozen=True)
class TaskActor:
    """A pose-bearing entity discovered through a stable task-attribute path."""

    key: str
    source: Any
    entity: Any
    entity_name: str | None
    shiftable: bool


def unwrap_entity(value):
    if (
        hasattr(value, "actor")
        and hasattr(value.actor, "get_pose")
        and hasattr(value.actor, "set_pose")
    ):
        return value.actor
    if hasattr(value, "get_pose") and hasattr(value, "set_pose"):
        return value
    return None


def is_shiftable_entity(task, path, entity):
    lowered_path = path.lower()
    if any(token in lowered_path for token in ("table", "wall", "ground", "camera", "light")):
        return False
    if hasattr(task, "robot") and entity.get_name() in getattr(task.robot, "gripper_name", []):
        return False
    try:
        position = np.asarray(entity.get_pose().p, dtype=np.float64)
    except Exception:
        return False
    return bool(
        -0.45 <= position[0] <= 0.45
        and -0.35 <= position[1] <= 0.35
        and 0.65 <= position[2] <= 1.25
    )


def discover_task_actors(task):
    """Discover unique pose-bearing entities reachable from task attributes."""
    actors = []
    seen = {}

    def visit(value, path, depth=0):
        if depth > 3 or value is None:
            return
        entity = unwrap_entity(value)
        if entity is not None:
            identity = id(entity)
            shiftable = is_shiftable_entity(task, path, entity)
            existing_index = seen.get(identity)
            if existing_index is None:
                try:
                    entity_name = entity.get_name()
                except Exception:
                    entity_name = None
                seen[identity] = len(actors)
                actors.append(
                    TaskActor(path, value, entity, entity_name, shiftable)
                )
            elif shiftable and not actors[existing_index].shiftable:
                existing = actors[existing_index]
                actors[existing_index] = TaskActor(
                    path, value, entity, existing.entity_name, True
                )
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", depth + 1)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}", depth + 1)

    for name, value in vars(task).items():
        if name.startswith("_") or name in _SKIP_TASK_ROOTS:
            continue
        visit(value, name)
    return actors
