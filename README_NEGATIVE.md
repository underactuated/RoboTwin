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

## Experimental Automatic Failure Detection

There is also a newer experimental collector that builds on
`replay_object_shift` and estimates the failure onset automatically:

- `collect_negative_data_failure_detection.sh`
- `script/collect_negative_data_failure_detection.py`

This path is intended for developing and validating failure-aware negative
datasets. It still saves standard RoboTwin HDF5/video episodes, but it also
saves replay pose traces and runs a detector that separates mostly-successful
prefix frames from clearly failed frames.

Example:

```bash
python script/collect_negative_data_failure_detection.py \
  hanging_mug my_config \
  --episode-num 3 \
  --save-setting my_config_auto_detection \
  --target-positive-traces 5 \
  --positive-attempt-budget 20
```

To also create a combined replay/detection video for each accepted episode:

```bash
python script/collect_negative_data_failure_detection.py \
  hanging_mug my_config \
  --episode-num 3 \
  --save-setting my_config_auto_detection \
  --create-detection-video
```

This experimental collector should normally use a distinct `--save-setting`
while it is being validated. That avoids mixing test outputs with a final
training dataset.

### How Automatic Detection Works

For each candidate source plan, the collector:

1. Generates one successful source plan lazily.
2. Replays it with zero perturbation. If the amplitude-0 replay fails, the
   source plan is rejected because it is not a reliable baseline.
3. Selects one shiftable actor and one shift direction, then escalates the
   replay-object-shift amplitude until the replay fails.
4. Collects successful replay traces near the success/failure boundary. These
   traces define the positive envelope.
5. Records the first failed replay as the negative episode.
6. Compares movable actor positions in the failed trace against the positive
   envelope and saves the detected failure onset.

The same selected actor and shift direction are kept during escalation. This
avoids changing the perturbation source while searching for the failure
amplitude.

By default, pose traces include shiftable/movable task actors only. For example,
for `hanging_mug` this usually tracks actors such as `mug` and `rack`, not fixed
background fixtures like the table or wall. Use `--include-nonshiftable-actors`
only when debugging actor discovery or scene setup.

The detector currently uses actor position components only (`x`, `y`, `z`):

- successful traces define per-frame mean and standard deviation;
- the failed trace is converted to normalized actor deviations;
- an actor is considered divergent when its deviation exceeds a threshold for
  `--persistence-frames` consecutive frames;
- the threshold is also adjusted by the early-frame baseline via
  `--initial-margin`, which prevents harmless initial replay offsets from
  triggering immediately.

Important detector parameters:

```text
--std-floor-m 0.005          minimum position std used for normalization
--threshold-k 3.0            base normalized-deviation threshold
--warmup-frames 3            early frames used for baseline adjustment
--initial-margin 1.1         multiplier for early-frame baseline adjustment
--persistence-frames 3       consecutive threshold crossings required
```

Important adaptive sampling parameters:

```text
--target-positive-traces 5       desired successful traces per failure
--min-positive-traces 3          reject failure if fewer positives are found
--positive-attempt-budget 20     maximum extra probes around the boundary
--backoff-factor 0.5             amplitude reduction factor during backoff
--min-probe-amplitude-ratio 0.01 stop probing below this fraction of base amplitude
```

Expected console messages include:

```text
replay shift target: rack (entity = 040_rack, base delta = [...])
negative replay still succeeded; escalating shift (...)
positive-envelope replay accepted (5/5, amplitude = ...)
automatic failure detection: episode = episode_000, onset = 294, actor = mug, confidence = normal
```

Some rejections are expected:

```text
source plan rejected because amplitude-0 replay failed
failure candidate rejected: only 2/3 minimum positive traces were available
```

These candidates are skipped and should not be saved as accepted negative
episodes.

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

Episode ids and simulator seeds are related but not identical. Files such as
`data/episode0.hdf5` and `video/episode0.mp4` use the accepted output episode
index. `seed.txt` stores the simulator seeds used to create source plans. In a
simple run with no rejected plans, these may both look like `0, 1, 2, ...`, but
they diverge as soon as any seed is rejected. The authoritative seed for an
episode is stored in `negative_info.json` under that episode record.

To start a run from a different simulator seed range:

```bash
python script/collect_negative_data.py hanging_mug my_config \
  --negative-mode replay_object_shift \
  --perturbation-amplitude 0.25 \
  --replay-shift-growth-factor 2.0 \
  --replay-shift-max-amplitude 8.0 \
  --save-setting my_config_negative_seed10000 \
  --seed-start 10000
```

If `seed.txt` already exists in the output directory, the collector resumes from
the last saved seed plus one. For a genuinely new seed range, use a new
`--save-setting` or remove the old output directory intentionally.

## Multi-GPU Batch Launcher

For larger negative-data generation runs, use:

- `script/launch_negative_data_multi_gpu.py`

This is a thin scheduler around `script/collect_negative_data.py`. It does not
implement a new collector. Instead, it launches multiple independent collector
subprocesses and assigns at most one subprocess to each selected GPU by setting
`CUDA_VISIBLE_DEVICES`.

The launcher is useful when generating negative samples for multiple RoboTwin
tasks. Each task is written to its own output directory:

```text
<output_root>/<task>/<save_setting>/
```

Do not use multiple concurrent processes writing to the same
`<output_root>/<task>/<save_setting>/` directory. The collector does not use
file locks for `seed.txt`, `negative_info.json`, episode indices, or videos.

### Editable Defaults

Common defaults are intentionally hardcoded near the top of
`script/launch_negative_data_multi_gpu.py`:

```python
DEFAULT_TASKS = [
    "hanging_mug",
]
DEFAULT_TASK_CONFIG = "my_config"
DEFAULT_EPISODES_PER_TASK = 30
DEFAULT_OUTPUT_ROOT = "./data"
DEFAULT_MAX_GPUS = 1
DEFAULT_GPU_IDS = None
DEFAULT_SEED_START = 0
DEFAULT_SEED_STRIDE_PER_TASK = 0

DEFAULT_NEGATIVE_MODE = "replay_object_shift"
DEFAULT_PERTURBATION_AMPLITUDE = 0.25
DEFAULT_REPLAY_SHIFT_GROWTH_FACTOR = 2.0
DEFAULT_REPLAY_SHIFT_MAX_AMPLITUDE = 8.0
```

For routine use, edit these values directly. CLI flags can override them for a
single run.

### Dry Run

Always start with a dry run:

```bash
python script/launch_negative_data_multi_gpu.py \
  --dry-run \
  --tasks hanging_mug beat_block_hammer \
  --episodes-per-task 20 \
  --output-root /data/sergey/robotwin_negative_runs \
  --max-gpus 2
```

The dry run prints:

- selected GPUs;
- generated per-task config names;
- output directories;
- log paths;
- exact collector commands.

Dry run does not launch collection jobs and does not write the temporary
per-job task config files.

### Launch a Batch

Example using GPUs `0` and `1`:

```bash
python script/launch_negative_data_multi_gpu.py \
  --tasks hanging_mug beat_block_hammer \
  --episodes-per-task 20 \
  --output-root /data/sergey/robotwin_negative_runs \
  --seed-start 10000 \
  --seed-stride-per-task 1000 \
  --max-gpus 2
```

Example with explicit GPU ids:

```bash
python script/launch_negative_data_multi_gpu.py \
  --tasks hanging_mug beat_block_hammer \
  --episodes-per-task 20 \
  --output-root /data/sergey/robotwin_negative_runs \
  --seed-start 10000 \
  --gpus 0 2
```

The launcher keeps a queue of tasks. When one subprocess exits, that GPU is
assigned the next pending task.

Logs are saved under:

```text
logs/negative_collection/
```

### Seed Ranges

The launcher forwards `--seed-start` to each `collect_negative_data.py`
subprocess. Use this when generating another batch and you want a different
simulator seed range:

```bash
python script/launch_negative_data_multi_gpu.py \
  --tasks hanging_mug beat_block_hammer \
  --episodes-per-task 20 \
  --output-root /data/sergey/robotwin_negative_runs \
  --seed-start 10000 \
  --max-gpus 2
```

If all tasks can reuse the same numeric seed range independently, leave
`--seed-stride-per-task` at `0`. This is usually fine because different tasks
write separate datasets and seed `10000` for `hanging_mug` is not the same
sample as seed `10000` for `beat_block_hammer`.

If you want globally non-overlapping numeric seed ranges across tasks, use a
stride larger than the expected number of tried seeds per task:

```bash
python script/launch_negative_data_multi_gpu.py \
  --tasks hanging_mug beat_block_hammer place_empty_cup \
  --episodes-per-task 100 \
  --seed-start 10000 \
  --seed-stride-per-task 10000 \
  --max-gpus 3
```

This assigns starts like:

```text
hanging_mug       -> 10000
beat_block_hammer -> 20000
place_empty_cup   -> 30000
```

Use a new `--save-setting-template` or a new `--output-root` for separate
batches. If a previous output directory already has `seed.txt`, the collector
resumes from that file instead of starting over from `--seed-start`.

### Output Root and Temporary Configs

`collect_negative_data.py` reads `episode_num` and `save_path` from the task
YAML config. To avoid modifying your base config, the launcher creates one
temporary config per job under `task_config/`, overriding only:

```yaml
episode_num: <episodes_per_task>
save_path: <output_root>
```

The generated config names look like:

```text
task_config/negative_batch_<run_id>_<base_config>_<task>.yml
```

By default these generated configs are kept, because they make it easier to
inspect or reproduce a generated dataset. To remove them automatically after
successful jobs, pass:

```bash
--cleanup-job-configs
```

### Save Setting Names

By default, each task writes to a save-setting name based on:

```text
{config}_negative_{mode}_amp{amplitude}
```

For example:

```text
/data/sergey/robotwin_negative_runs/hanging_mug/my_config_negative_replay_object_shift_amp0.25/
```

You can override the naming pattern:

```bash
python script/launch_negative_data_multi_gpu.py \
  --tasks hanging_mug beat_block_hammer \
  --save-setting-template "{config}_negative_batch_{run_id}" \
  --output-root /data/sergey/robotwin_negative_runs \
  --max-gpus 2
```

Available template fields:

```text
{task}
{config}
{mode}
{amplitude}
{run_id}
```

### Negative-Generation Parameters

The launcher forwards these parameters to `collect_negative_data.py`:

```bash
--negative-mode replay_object_shift
--perturbation-amplitude 0.25
--replay-shift-growth-factor 2.0
--replay-shift-max-amplitude 8.0
```

These can be changed either in the editable defaults block or with CLI flags.

Current limitation: this launcher targets the regular negative collector,
`script/collect_negative_data.py`. It does not currently launch the experimental
automatic failure-detection collector,
`script/collect_negative_data_failure_detection.py`.

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

The automatic failure-detection collector additionally writes debug and
detection outputs:

```text
data/<task>/<setting>/
  failure_detection_debug/
    episode_000/
      dry_attempt_000_amp0_success.npz
      dry_attempt_001_amp0.25_success.npz
      dry_attempt_004_amp2_failure.npz
      recorded_attempt_004_amp2_failure.npz
      failure_detection.json
      failure_detection_stats.npz
      failure_detection_score.png
      failure_detection_video.mp4        # only when requested
```

Each trace `.npz` contains:

- `poses`: array shaped `(frames, actors, 7)` storing
  `[x, y, z, qw, qx, qy, qz]`;
- `actor_keys`: actor names in the same order as the second `poses` dimension;
- `final_success`: whether that replay succeeded;
- replay metadata such as amplitude, attempt, and seed.

`failure_detection.json` is the compact summary to inspect first. It includes
the positive traces used, failed trace, actor thresholds, detected onset frame,
dominant actor at onset, score at onset, and confidence level.

`failure_detection_stats.npz` stores the full time-series data used by the
plots and detection video.

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

In the automatic failure-detection collector, amplitude-0 replay failure is a
special rejection. It means the supposedly successful source plan did not replay
cleanly even without object shifting. The collector rejects that source seed and
tries another one, because such a trace cannot define a trustworthy
success/failure boundary.

## Trace Visualization and Offline Analysis

Pose traces can be plotted independently from collection:

```bash
python script/visualize_pose_traces.py \
  data/hanging_mug/my_config_auto_detection/failure_detection_debug/episode_000 \
  --actors mug rack \
  --components x y z \
  --aggregate-positives
```

By default this saves:

```text
<trace_dir>/pose_traces.png
```

Useful options:

```text
--actors mug rack                 plot only selected actors
--components x y z                plot only selected pose components
--aggregate-positives             show successful traces as mean ± std
--run-kinds dry recorded          include dry and recorded traces
--output /path/to/figure.png      choose output path
```

Failure detection can also be rerun offline on an existing trace directory:

```bash
python script/analyze_replay_failure_detection.py \
  data/hanging_mug/my_config_auto_detection/failure_detection_debug/episode_000
```

This rewrites:

```text
failure_detection.json
failure_detection_stats.npz
failure_detection_score.png
```

You can override detector parameters when rerunning:

```bash
python script/analyze_replay_failure_detection.py \
  data/hanging_mug/my_config_auto_detection/failure_detection_debug/episode_000 \
  --threshold-k 3.0 \
  --persistence-frames 3
```

To create the combined replay/detection video after collection:

```bash
python script/create_failure_detection_video.py \
  data/hanging_mug/my_config_auto_detection \
  --episode 0
```

Default output:

```text
data/hanging_mug/my_config_auto_detection/failure_detection_debug/episode_000/failure_detection_video.mp4
```

The video uses the normal episode replay video on top and a threshold-ratio
strip below it. The strip shows the current frame cursor, threshold line,
detected onset frame, post-onset shading, current score, and actor with the
largest normalized deviation.

`matplotlib` is required for static plots. `opencv-python`/`cv2` is required for
combined videos.

## Implementation Summary

`planner_perturb` wraps shared task primitives at runtime:

- `get_grasp_pose`
- `place_actor`
- `move_to_pose`
- `move_by_displacement`
- `set_gripper`

`replay_object_shift` scans task attributes for actor-like objects, randomly selects movable candidates, shifts up to `replay_shift_max_objects` objects, and replays the original successful joint path.

The actor selection is intentionally generic. It does not currently distinguish manipulated objects from target/fixture objects; it filters obvious non-task objects such as robot, scene, cameras, table, wall, ground, and objects outside the table workspace.

The automatic failure-detection collector reuses the same replay-shift machinery
but records actor poses at the dataset frame cadence during dry and recorded
replays. It then compares the failed replay against successful boundary traces
for all tracked movable actors, instead of requiring a manually selected actor.
