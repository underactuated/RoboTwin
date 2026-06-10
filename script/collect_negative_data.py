import sys

sys.path.append("./")

import importlib
import json
import math
import os
import time
from argparse import ArgumentParser

import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import yaml

from envs import *


PLANNER_PERTURB = "planner_perturb"
REPLAY_OBJECT_SHIFT = "replay_object_shift"
NEGATIVE_MODES = (PLANNER_PERTURB, REPLAY_OBJECT_SHIFT)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


class ContinuousPerturber:
    def __init__(self, amplitude, seed):
        self.amplitude = float(amplitude)
        self.rng = np.random.default_rng(seed)
        self.events = []
        self.xyz_scale = 0.03
        self.rot_scale = math.radians(5)
        self.displacement_scale = 0.03
        self.scalar_scale = 0.03
        self.gripper_scale = 0.15

    @property
    def enabled(self):
        return self.amplitude > 0

    def normal(self, scale, size=None):
        return self.rng.normal(0.0, self.amplitude * scale, size=size)

    def record(self, name, value):
        self.events.append({"name": name, "value": value})

    def perturb_pose(self, pose, name):
        if not self.enabled or pose is None:
            return pose

        original_type = type(pose)
        if isinstance(pose, sapien.Pose):
            pose_list = pose.p.tolist() + pose.q.tolist()
        elif isinstance(pose, np.ndarray):
            pose_list = pose.tolist()
        else:
            pose_list = list(pose)

        if len(pose_list) not in (3, 7):
            return pose

        xyz_delta = self.normal(self.xyz_scale, size=3)
        pose_arr = np.asarray(pose_list, dtype=np.float64)
        pose_arr[:3] += xyz_delta

        event = {"xyz_delta": xyz_delta.tolist()}
        if len(pose_arr) == 7:
            axis = self.rng.normal(size=3)
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = axis / axis_norm
            angle = float(self.normal(self.rot_scale))
            delta_q = t3d.quaternions.axangle2quat(axis, angle)
            pose_arr[3:] = normalize_quat(t3d.quaternions.qmult(delta_q, pose_arr[3:]))
            event["rot_axis"] = axis.tolist()
            event["rot_angle"] = angle

        self.record(name, event)
        if original_type is np.ndarray:
            return pose_arr
        return pose_arr.tolist()

    def perturb_scalar(self, value, name, scale, low=None, high=None):
        if not self.enabled or value is None:
            return value
        perturbed = float(value + self.normal(scale))
        if low is not None or high is not None:
            perturbed = float(np.clip(
                perturbed,
                -np.inf if low is None else low,
                np.inf if high is None else high,
            ))
        self.record(name, {"delta": perturbed - float(value), "result": perturbed})
        return perturbed

    @staticmethod
    def restore_task(task):
        for method_name in (
            "get_grasp_pose",
            "place_actor",
            "move_to_pose",
            "move_by_displacement",
            "set_gripper",
        ):
            method = getattr(task.__class__, method_name).__get__(task, task.__class__)
            setattr(task, method_name, method)

    def patch_task(self, task):
        self.restore_task(task)
        if not self.enabled:
            return

        original_get_grasp_pose = task.get_grasp_pose
        original_place_actor = task.place_actor
        original_move_to_pose = task.move_to_pose
        original_move_by_displacement = task.move_by_displacement
        original_set_gripper = task.set_gripper

        def get_grasp_pose_wrapper(*args, **kwargs):
            pose = original_get_grasp_pose(*args, **kwargs)
            return self.perturb_pose(pose, "get_grasp_pose")

        def place_actor_wrapper(actor, arm_tag, target_pose, *args, **kwargs):
            target_pose = self.perturb_pose(target_pose, "place_actor.target_pose")
            if "pre_dis" in kwargs:
                kwargs["pre_dis"] = self.perturb_scalar(
                    kwargs["pre_dis"], "place_actor.pre_dis", self.scalar_scale, low=0.0)
            if "dis" in kwargs:
                kwargs["dis"] = self.perturb_scalar(kwargs["dis"], "place_actor.dis", self.scalar_scale, low=0.0)
            return original_place_actor(actor, arm_tag, target_pose, *args, **kwargs)

        def move_to_pose_wrapper(arm_tag, target_pose):
            return original_move_to_pose(arm_tag, self.perturb_pose(target_pose, "move_to_pose.target_pose"))

        def move_by_displacement_wrapper(
            arm_tag,
            x=0.0,
            y=0.0,
            z=0.0,
            quat=None,
            move_axis="world",
        ):
            x = self.perturb_scalar(x, "move_by_displacement.x", self.displacement_scale)
            y = self.perturb_scalar(y, "move_by_displacement.y", self.displacement_scale)
            z = self.perturb_scalar(z, "move_by_displacement.z", self.displacement_scale)
            if quat is not None:
                quat = self.perturb_pose([0.0, 0.0, 0.0] + list(quat), "move_by_displacement.quat")[3:]
            return original_move_by_displacement(arm_tag, x=x, y=y, z=z, quat=quat, move_axis=move_axis)

        def set_gripper_wrapper(set_tag="together", left_pos=None, right_pos=None):
            left_pos = self.perturb_scalar(left_pos, "set_gripper.left_pos", self.gripper_scale, low=0.0, high=1.0)
            right_pos = self.perturb_scalar(right_pos, "set_gripper.right_pos", self.gripper_scale, low=0.0, high=1.0)
            return original_set_gripper(set_tag=set_tag, left_pos=left_pos, right_pos=right_pos)

        task.get_grasp_pose = get_grasp_pose_wrapper
        task.place_actor = place_actor_wrapper
        task.move_to_pose = move_to_pose_wrapper
        task.move_by_displacement = move_by_displacement_wrapper
        task.set_gripper = set_gripper_wrapper


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
        for path, obj, entity in candidates:
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
        candidates = []
        seen = set()
        skip_roots = {
            "engine", "renderer", "scene", "viewer", "robot", "cameras", "camera",
            "info", "now_obs", "world_pcd", "cluttered_objs", "record_cluttered_objects",
            "direction_light_lst", "point_light_lst", "file_path", "prohibited_area",
        }

        def visit(value, path, depth=0):
            if depth > 3:
                return
            if value is None:
                return
            entity = self.unwrap_entity(value)
            if entity is not None:
                key = id(entity)
                if key not in seen and self.is_shiftable_entity(task, path, entity):
                    seen.add(key)
                    candidates.append((path, value, entity))
                return
            if isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    visit(item, f"{path}[{idx}]", depth + 1)
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(item, f"{path}.{key}", depth + 1)

        for name, value in vars(task).items():
            if name.startswith("_") or name in skip_roots:
                continue
            visit(value, name)
        return candidates

    @staticmethod
    def unwrap_entity(value):
        if hasattr(value, "actor") and hasattr(value.actor, "get_pose") and hasattr(value.actor, "set_pose"):
            return value.actor
        if hasattr(value, "get_pose") and hasattr(value, "set_pose"):
            return value
        return None

    @staticmethod
    def is_shiftable_entity(task, path, entity):
        lowered_path = path.lower()
        if any(token in lowered_path for token in ("table", "wall", "ground", "camera", "light")):
            return False
        if hasattr(task, "robot") and entity.get_name() in getattr(task.robot, "gripper_name", []):
            return False
        try:
            pose = entity.get_pose()
        except Exception:
            return False
        p = np.asarray(pose.p, dtype=np.float64)
        if not (-0.45 <= p[0] <= 0.45 and -0.35 <= p[1] <= 0.35 and 0.65 <= p[2] <= 1.25):
            return False
        return True

    def shift_entity(self, path, entity):
        pose = entity.get_pose()
        old_p = np.asarray(pose.p, dtype=np.float64)
        delta = np.array([
            self.normal(self.xy_scale),
            self.normal(self.xy_scale),
            self.normal(self.z_scale),
        ], dtype=np.float64)

        # Keep arm-selection branches stable for scripts that choose left/right by x sign.
        new_x = old_p[0] + delta[0]
        if old_p[0] != 0 and old_p[0] * new_x < 0:
            delta[0] = -0.5 * old_p[0]

        new_p = old_p + delta
        new_p[0] = np.clip(new_p[0], -0.42, 0.42)
        new_p[1] = np.clip(new_p[1], -0.32, 0.32)
        new_pose = sapien.Pose(new_p, pose.q)
        try:
            entity.set_pose(new_pose)
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


def prepare_args(
    task_name,
    task_config,
    perturbation_amplitude,
    negative_mode,
    replay_shift_growth_factor=None,
    replay_shift_max_amplitude=None,
    save_setting=None,
):
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["negative_mode"] = negative_mode
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment):
        robot_file = embodiment_types[embodiment]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    else:
        raise RuntimeError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = embodiment_name
    args["task_config"] = task_config
    args["perturbation_amplitude"] = float(
        args.get("perturbation_amplitude", 0.0) if perturbation_amplitude is None else perturbation_amplitude)
    if replay_shift_growth_factor is not None:
        args["replay_shift_growth_factor"] = replay_shift_growth_factor
    if replay_shift_max_amplitude is not None:
        args["replay_shift_max_amplitude"] = replay_shift_max_amplitude
    if save_setting is not None:
        args["save_setting"] = save_setting
    elif negative_mode == PLANNER_PERTURB:
        args["save_setting"] = f"{args['task_config']}_negative_amp{args['perturbation_amplitude']:g}"
    else:
        args["save_setting"] = f"{args['task_config']}_negative_{negative_mode}_amp{args['perturbation_amplitude']:g}"
    args["save_path"] = os.path.join(
        args["save_path"],
        str(args["task_name"]),
        args["save_setting"],
    )
    return args


def write_seed_file(save_path, seed_list):
    with open(os.path.join(save_path, "seed.txt"), "w") as file:
        for seed in seed_list:
            file.write("%s " % seed)


def load_seed_file(save_path):
    with open(os.path.join(save_path, "seed.txt"), "r") as file:
        seed_list = file.read().split()
    return [int(i) for i in seed_list]


def ensure_negative_task_config(args):
    source_config_path = os.path.join("task_config", f"{args['task_config']}.yml")
    target_config_path = os.path.join("task_config", f"{args['save_setting']}.yml")
    if os.path.exists(target_config_path):
        return

    with open(source_config_path, "r", encoding="utf-8") as file:
        config = yaml.load(file.read(), Loader=yaml.FullLoader)

    config["save_path"] = os.path.dirname(os.path.dirname(args["save_path"]))
    config["perturbation_amplitude"] = args["perturbation_amplitude"]
    config["negative_mode"] = args["negative_mode"]
    if "replay_shift_growth_factor" in args:
        config["replay_shift_growth_factor"] = args["replay_shift_growth_factor"]
    if "replay_shift_max_amplitude" in args:
        config["replay_shift_max_amplitude"] = args["replay_shift_max_amplitude"]
    config["source_task_config"] = args["task_config"]

    with open(target_config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def ensure_scene_info(args):
    scene_info_path = os.path.join(args["save_path"], "scene_info.json")
    if os.path.exists(scene_info_path):
        return

    negative_info_path = os.path.join(args["save_path"], "negative_info.json")
    if not os.path.exists(negative_info_path):
        return

    with open(negative_info_path, "r", encoding="utf-8") as file:
        negative_info = json.load(file)

    scene_info = {}
    for episode_key, episode_data in negative_info.items():
        scene_info[episode_key] = episode_data.get("task_info", {})

    with open(scene_info_path, "w", encoding="utf-8") as file:
        json.dump(scene_info, file, ensure_ascii=False, indent=4)


def collect_plans(task, args, accept_success, target_num=None):
    target_num = args["episode_num"] if target_num is None else target_num
    epid, plan_num, reject_num, seed_list = 0, 0, 0, []
    os.makedirs(args["save_path"], exist_ok=True)
    args["need_plan"] = True

    seed_path = os.path.join(args["save_path"], "seed.txt")
    if os.path.exists(seed_path):
        seed_list = load_seed_file(args["save_path"])
        if seed_list:
            plan_num = len(seed_list)
            epid = seed_list[-1] + 1
        print(f"Exist seed file, Start from: {epid} / {plan_num}")

    max_tries = int(args.get("negative_max_tries", max(target_num, args["episode_num"]) * 50))
    while plan_num < target_num and epid < max_tries:
        perturber = None
        try:
            task.setup_demo(now_ep_num=plan_num, seed=epid, **args)
            if args["negative_mode"] == PLANNER_PERTURB:
                perturber = ContinuousPerturber(args["perturbation_amplitude"], seed=(epid + 1000003))
                perturber.patch_task(task)
            else:
                ContinuousPerturber.restore_task(task)

            info = task.play_once()
            final_success = bool(task.plan_success and task.check_success())
            has_motion = len(task.left_joint_path) > 0 or len(task.right_joint_path) > 0
            accept = has_motion and (final_success if accept_success else not final_success)

            if accept:
                status = "success plan" if accept_success else "negative plan"
                print(f"{status} episode {plan_num} accepted! (seed = {epid})")
                task.save_traj_data(plan_num)
                if not accept_success:
                    save_negative_metadata(args, plan_num, epid, final_success, perturber, info)
                seed_list.append(epid)
                plan_num += 1
            else:
                print(f"episode {plan_num} rejected! (seed = {epid}, success = {final_success})")
                reject_num += 1

            task.close_env()
            if args["render_freq"]:
                task.viewer.close()
        except UnStableError as e:
            print(f"episode {plan_num} rejected as unstable! (seed = {epid})")
            print("Error: ", e)
            reject_num += 1
            task.close_env()
            if args["render_freq"]:
                task.viewer.close()
            time.sleep(0.3)
        except Exception as e:
            print(f"episode {plan_num} rejected by exception! (seed = {epid})")
            print("Error: ", e)
            reject_num += 1
            task.close_env()
            if args["render_freq"]:
                task.viewer.close()
            time.sleep(1)

        epid += 1
        write_seed_file(args["save_path"], seed_list)

    if plan_num < target_num:
        raise RuntimeError(
            f"Only collected {plan_num} plans after {epid} tries. "
            f"Try increasing perturbation_amplitude or negative_max_tries.")
    print(f"\nComplete planning, rejected {reject_num} times / {epid} tries\n")


def save_negative_metadata(args, episode_idx, seed, final_success, perturber, info, extra=None):
    metadata_path = os.path.join(args["save_path"], "negative_info.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    else:
        metadata = {}

    episode_key = f"episode_{episode_idx}"
    previous = metadata.get(episode_key, {})
    perturbation_events = previous.get("perturbation_events", [])
    if perturber is not None:
        perturbation_events = perturber.events

    record = {
        "seed": seed,
        "negative_mode": args["negative_mode"],
        "perturbation_amplitude": args["perturbation_amplitude"],
        "final_success": final_success,
        "perturbation_events": perturbation_events,
        "task_info": info,
    }
    if extra:
        record.update(extra)
    metadata[episode_key] = record

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=4)


def prepare_replay_episode(task, args, episode_idx, seed, traj_idx):
    task.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
    ContinuousPerturber.restore_task(task)

    traj_data = task.load_tran_data(traj_idx)
    args["left_joint_path"] = traj_data["left_joint_path"]
    args["right_joint_path"] = traj_data["right_joint_path"]
    task.set_path_lst(args)

    extra = None
    if args["negative_mode"] == REPLAY_OBJECT_SHIFT:
        actual_amplitude = args.get("actual_perturbation_amplitude", args["perturbation_amplitude"])
        shifter = ReplayObjectShifter(
            actual_amplitude,
            seed=seed + 2000003,
            max_objects=args.get("replay_shift_max_objects", 1),
        )
        shift_events = shifter.apply(task)
        extra = {
            "perturbation_events": shift_events,
            "actual_perturbation_amplitude": actual_amplitude,
            "replay_shift_attempt": args.get("replay_shift_attempt"),
        }
    return extra


def run_replay_once(task, args, episode_idx, seed, traj_idx, save_data):
    args["need_plan"] = False
    args["render_freq"] = 0
    args["save_data"] = save_data
    task.save_data = save_data
    extra = prepare_replay_episode(task, args, episode_idx, seed, traj_idx)
    info = task.play_once()
    final_success = bool(task.plan_success and task.check_success())
    return final_success, info, extra


def collect_negative_data(task, args):
    if not args["collect_data"]:
        return

    print("\033[93m" + "[Start Negative Data Collection]" + "\033[0m")
    seed_list = load_seed_file(args["save_path"])
    clear_cache_freq = args["clear_cache_freq"]

    def exist_hdf5(idx):
        file_path = os.path.join(args["save_path"], "data", f"episode{idx}.hdf5")
        return os.path.exists(file_path)

    episode_idx = 0
    while exist_hdf5(episode_idx):
        episode_idx += 1

    seed_cursor = episode_idx
    replay_reject_num = 0
    max_replay_tries = int(args.get("negative_max_replay_tries", args["episode_num"] * 50))
    base_amplitude = float(args["perturbation_amplitude"])
    growth_factor = float(args.get("replay_shift_growth_factor", 2.0))
    max_amplitude = float(args.get("replay_shift_max_amplitude", max(base_amplitude, 8.0)))
    if args["negative_mode"] == REPLAY_OBJECT_SHIFT and base_amplitude <= 0.0:
        raise ValueError("perturbation_amplitude must be positive for replay_object_shift")
    if args["negative_mode"] == REPLAY_OBJECT_SHIFT and growth_factor <= 1.0:
        raise ValueError("replay_shift_growth_factor must be greater than 1.0")

    while episode_idx < args["episode_num"]:
        if seed_cursor >= len(seed_list):
            collect_plans(task, args, accept_success=True, target_num=seed_cursor + 1)
            seed_list = load_seed_file(args["save_path"])
        if replay_reject_num >= max_replay_tries:
            raise RuntimeError(
                f"Only collected {episode_idx} negative replays after {replay_reject_num} rejected replays. "
                "Try increasing perturbation_amplitude or replay_shift_max_objects.")

        seed = seed_list[seed_cursor]
        traj_idx = seed_cursor if args["negative_mode"] == REPLAY_OBJECT_SHIFT else episode_idx
        print(f"\033[34mNegative task name: {args['task_name']} seed: {seed}\033[0m")

        if args["negative_mode"] == REPLAY_OBJECT_SHIFT:
            actual_amplitude = base_amplitude
            replay_attempt = 0
            dry_failure_found = False

            while True:
                replay_attempt += 1
                args["actual_perturbation_amplitude"] = actual_amplitude
                args["replay_shift_attempt"] = replay_attempt
                try:
                    final_success, _, _ = run_replay_once(
                        task,
                        args,
                        episode_idx=episode_idx,
                        seed=seed,
                        traj_idx=traj_idx,
                        save_data=False,
                    )
                except Exception as e:
                    task.close_env(clear_cache=False)
                    replay_reject_num += 1
                    print(
                        f"negative replay dry run failed to construct! "
                        f"(seed = {seed}, episode = {episode_idx}, "
                        f"amplitude = {actual_amplitude:g}, error = {e})")
                    if actual_amplitude >= max_amplitude:
                        break
                    actual_amplitude = min(actual_amplitude * growth_factor, max_amplitude)
                    continue

                task.close_env(clear_cache=False)
                if not final_success:
                    dry_failure_found = True
                    break

                replay_reject_num += 1
                print(
                    f"negative replay still succeeded; escalating shift "
                    f"(seed = {seed}, episode = {episode_idx}, amplitude = {actual_amplitude:g})")
                if actual_amplitude >= max_amplitude:
                    break
                actual_amplitude = min(actual_amplitude * growth_factor, max_amplitude)

            if not dry_failure_found:
                seed_cursor += 1
                args.pop("actual_perturbation_amplitude", None)
                args.pop("replay_shift_attempt", None)
                continue

            args["actual_perturbation_amplitude"] = actual_amplitude
            args["replay_shift_attempt"] = replay_attempt

        try:
            final_success, info, extra = run_replay_once(
                task,
                args,
                episode_idx=episode_idx,
                seed=seed,
                traj_idx=traj_idx,
                save_data=True,
            )
        except Exception as e:
            task.close_env(clear_cache=False)
            if args["negative_mode"] == REPLAY_OBJECT_SHIFT:
                task.remove_data_cache()
                replay_reject_num += 1
                seed_cursor += 1
                args.pop("actual_perturbation_amplitude", None)
                args.pop("replay_shift_attempt", None)
                print(
                    f"negative replay rejected during recording rerun! "
                    f"(seed = {seed}, episode = {episode_idx}, error = {e})")
                continue
            raise
        save_negative_metadata(args, episode_idx, seed, final_success, None, info, extra=extra)

        task.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
        if final_success:
            task.remove_data_cache()
            if args["negative_mode"] == REPLAY_OBJECT_SHIFT:
                replay_reject_num += 1
                seed_cursor += 1
                args.pop("actual_perturbation_amplitude", None)
                args.pop("replay_shift_attempt", None)
                print(
                    f"negative replay rejected after recording rerun! "
                    f"(seed = {seed}, episode = {episode_idx}, success = True)")
                continue
            raise RuntimeError(
                f"Negative replay unexpectedly succeeded for episode {episode_idx}. "
                "Try increasing perturbation_amplitude or replay_shift_max_objects.")
        task.merge_pkl_to_hdf5_video()
        task.remove_data_cache()
        episode_idx += 1
        seed_cursor += 1
        args.pop("actual_perturbation_amplitude", None)
        args.pop("replay_shift_attempt", None)

    ensure_negative_task_config(args)
    ensure_scene_info(args)
    command = (
        f"cd description && bash gen_episode_instructions.sh "
        f"{args['task_name']} {args['save_setting']} {args['language_num']}")
    os.system(command)


def main(
    task_name=None,
    task_config=None,
    perturbation_amplitude=None,
    negative_mode=PLANNER_PERTURB,
    replay_shift_growth_factor=None,
    replay_shift_max_amplitude=None,
    save_setting=None,
):
    task = class_decorator(task_name)
    args = prepare_args(
        task_name,
        task_config,
        perturbation_amplitude,
        negative_mode,
        replay_shift_growth_factor=replay_shift_growth_factor,
        replay_shift_max_amplitude=replay_shift_max_amplitude,
        save_setting=save_setting,
    )
    print("============= Negative Collection Config =============\n")
    print("\033[95mNegative Mode:\033[0m " + str(args["negative_mode"]))
    print("\033[95mPerturbation Amplitude:\033[0m " + str(args["perturbation_amplitude"]))
    if args["negative_mode"] == REPLAY_OBJECT_SHIFT:
        print("\033[95mReplay Shift Growth Factor:\033[0m " + str(args.get("replay_shift_growth_factor", 2.0)))
        print("\033[95mReplay Shift Max Amplitude:\033[0m " + str(args.get("replay_shift_max_amplitude", max(args["perturbation_amplitude"], 8.0))))
    print("\033[94mSave Path:\033[0m " + str(args["save_path"]))
    print("\n======================================================")
    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    if not args["use_seed"]:
        collect_plans(task, args, accept_success=(args["negative_mode"] == REPLAY_OBJECT_SHIFT))
    else:
        print("\033[93m" + "Use Saved Negative Seeds List".center(30, "-") + "\033[0m")

    collect_negative_data(task, args)


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "--perturbation-amplitude",
        type=float,
        default=None,
        help="Non-negative scalar. 0 recovers unperturbed planner/execution.",
    )
    parser.add_argument(
        "--negative-mode",
        choices=NEGATIVE_MODES,
        default=PLANNER_PERTURB,
        help="planner_perturb perturbs privileged subgoals; replay_object_shift replays successful plans after shifting an object.",
    )
    parser.add_argument(
        "--replay-shift-growth-factor",
        type=float,
        default=None,
        help="Multiplier used to escalate replay_object_shift amplitude after each successful dry replay.",
    )
    parser.add_argument(
        "--replay-shift-max-amplitude",
        type=float,
        default=None,
        help="Maximum actual amplitude tried by replay_object_shift escalation.",
    )
    parser.add_argument(
        "--save-setting",
        type=str,
        default=None,
        help="Override output config directory name under data/<task>. Use the original config name for packer compatibility.",
    )
    parsed = parser.parse_args()
    if parsed.perturbation_amplitude is not None and parsed.perturbation_amplitude < 0:
        raise ValueError("perturbation-amplitude must be non-negative")
    main(
        task_name=parsed.task_name,
        task_config=parsed.task_config,
        perturbation_amplitude=parsed.perturbation_amplitude,
        negative_mode=parsed.negative_mode,
        replay_shift_growth_factor=parsed.replay_shift_growth_factor,
        replay_shift_max_amplitude=parsed.replay_shift_max_amplitude,
        save_setting=parsed.save_setting,
    )
