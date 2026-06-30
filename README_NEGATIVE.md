# Negative Data Collection for RoboTwin

This repository contains a local extension for generating negative RoboTwin trajectories: simulated rollouts that are produced by the original scripted task controllers but do not reach task success. The goal is to create failure data for VLA/manipulation training, especially to expose models to states that are out of distribution relative to purely successful imitation data.

The extension is intentionally parallel to the original collection path. It keeps the original task files mostly untouched and adds a separate collector:

- `collect_negative_data.sh`
- `script/collect_negative_data.py`

## Main Idea

RoboTwin task controllers use privileged simulator state to choose scripted grasp/place/move targets. Negative collection perturbs this privileged execution while still using the same controllers and simulator data format.

Two negative modes are supported:

- `planner_perturb`: perturbs privileged target poses, displacements, rotations, and gripper commands while the controller is planning. This often creates short failed trajectories because RoboTwin aborts after a later subgoal becomes unplannable.
- `replay_object_shift`: first collects a successful unperturbed trajectory, then replays the full saved joint path after shifting a movable object in the scene. This usually produces longer post-failure trajectories because the robot continues executing the stale successful plan after the world has changed.

The default wrapper currently uses `replay_object_shift`.

## Basic Usage

The wrapper has the same calling convention as `collect_data.sh`:

```bash
bash collect_negative_data.sh <task_name> <task_config> <gpu_id>
```

Example:

```bash
bash collect_negative_data.sh beat_block_hammer my_config 0
```

The wrapper currently expands to:

```bash
python script/collect_negative_data.py "$task_name" "$task_config" \
  --negative-mode replay_object_shift \
  --perturbation-amplitude 0.25 \
  --replay-shift-growth-factor 2.0 \
  --replay-shift-max-amplitude 8.0 \
  --save-setting "$task_config"
```

`--save-setting "$task_config"` makes the negative data save under `data/<task>/<task_config>/`, matching the directory name expected by downstream RobotWin-to-LMDB conversion scripts that pass `--config_name <task_config>`.

## Perturbation Amplitude

`--perturbation-amplitude` is the base perturbation strength. In `replay_object_shift`, object shifts are roughly:

```text
0.03 meters * actual_perturbation_amplitude
```

For `replay_object_shift`, the collector performs a dry replay first with `save_data=False`. If the shifted replay still succeeds, it escalates the actual amplitude:

```text
base, base * growth_factor, base * growth_factor^2, ... up to max_amplitude
```

For example:

```bash
--perturbation-amplitude 0.25 \
--replay-shift-growth-factor 2.0 \
--replay-shift-max-amplitude 8.0
```

tries actual amplitudes:

```text
0.25, 0.5, 1.0, 2.0, 4.0, 8.0
```

The amplitude resets to the base value for the next output episode.

## Output Format

The collector writes standard RoboTwin-style episode files:

```text
data/<task>/<setting>/
  seed.txt
  scene_info.json
  negative_info.json
  data/episode0.hdf5
  video/episode0.mp4
  instructions/episode0.json
```

When `--save-setting "$task_config"` is used, `<setting>` is the original config name, such as `my_config`.

`negative_info.json` stores extra negative-collection metadata, including:

- `negative_mode`
- `perturbation_amplitude`
- `actual_perturbation_amplitude` for replay-shift episodes
- `replay_shift_attempt`
- `perturbation_events`
- `final_success`

`scene_info.json` is also written so the existing instruction-generation and packing pipeline can read the dataset normally.

## LMDB Conversion Compatibility

The RoboOrchard `robotwin_packer.py` validates directories by exact config name. If the packer is called with:

```bash
--config_name my_config
```

then it only accepts:

```text
data/<task>/my_config
```

and rejects names like:

```text
data/<task>/my_config_negative_replay_object_shift_amp0.25
```

For this reason, the current `collect_negative_data.sh` passes:

```bash
--save-setting "$task_config"
```

so the negative dataset is saved under the original config directory and can be converted by the existing shell script.

Important: do not mix positive and negative HDF5 episodes in the same `data/<task>/<task_config>/` folder unless that is intentional. For a clean negative LMDB run, start from an empty task/config data folder.

## Notes on Rejections

In `replay_object_shift`, not every shifted replay fails. The collector may reject candidates for several reasons:

- the shifted replay still succeeds during the dry run;
- the shifted scene makes a scripted target pose invalid;
- the recording rerun unexpectedly succeeds;
- the maximum replay-shift amplitude is reached without a usable failure.

Successful rejects are fast because they use dry replay without image/cache saving. Only accepted failures are replayed with `save_data=True` and merged into HDF5/video.

## Implementation Summary

`planner_perturb` wraps shared task primitives at runtime:

- `get_grasp_pose`
- `place_actor`
- `move_to_pose`
- `move_by_displacement`
- `set_gripper`

`replay_object_shift` scans task attributes for actor-like objects, randomly selects movable candidates, shifts up to `replay_shift_max_objects` objects, and replays the original successful joint path.

The actor selection is intentionally generic. It does not currently distinguish manipulated objects from target/fixture objects; it filters obvious non-task objects such as robot, scene, cameras, table, wall, ground, and objects outside the table workspace.
