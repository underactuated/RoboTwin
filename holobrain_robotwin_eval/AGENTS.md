# AGENTS.md

Purpose: fast handoff context for future Codex sessions working on HoloBrain RobotWin evaluation.

## Scope
- This file is specific to `projects/holobrain/holobrain_robotwin_eval`.
- Repo-wide Docker, CUDA, and dependency setup notes remain in the root `AGENTS.md`.

## Resolved Eval Setup (March 12, 2026)

### Initial Problem
- HoloBrain RobotWin eval failed when trying to combine RoboOrchardLab and RoboTwin.
- A shared environment was not desirable because it was large and fragile.
- Running with separate envs initially failed on missing imports such as `sapien` and `robo_orchard_core`.
- Exported models also failed to load during eval because the export layout stores model files under `model/`, while eval defaulted to `model_prefix=model`.
- `--test_num` passed to outer eval was ignored by RoboTwin's inner `script/eval_policy.py`, which hardcoded `test_num = 100`.

### Working Solution
- No extra installs are required in either existing environment.
- No third shared environment is required.
- Use a two-env launch pattern:
  - outer process: RoboOrchardLab env runs `projects/holobrain/holobrain_robotwin_eval/eval.py`
  - inner process: RoboTwin env is selected by prefixing `PATH`, so spawned `python3` resolves to RoboTwin's interpreter
- Expose both RoboOrchardLab source and its installed packages to the inner process:
  - `PYTHONPATH=/home/sergey/Documents/projects/RoboOrchardLab:/home/sergey/Documents/projects/RoboOrchardLab/.venv/lib/python3.10/site-packages`
- Eval must still be launched from repo root:
  - `/home/sergey/Documents/projects/RoboOrchardLab`
- `--model_config` should point to the exported model workspace root.
- Because export writes model files to `<export_dir>/model/`, eval must use:
  - `--model_prefix model/model`

### Modifications Made In This Repo
- `projects/holobrain/holobrain_robotwin_eval/eval.py`
  - inner launch changed from `python3 script/eval_policy.py` to `python3 holobrain_robotwin_eval/eval_policy_wrapper.py`
- `projects/holobrain/holobrain_robotwin_eval/eval_policy_wrapper.py`
  - added as a thin wrapper around RoboTwin eval logic
  - honors the outer `--test_num` value instead of RoboTwin's hardcoded `100`
  - imports `Sapien_TEST` from `script.test_render`
- Current behavior:
  - running with `--test_num 10` now performs 10 valid eval seeds per task

### Export Notes
- `projects/holobrain/export.py` writes an export workspace; it does not read a training workspace by default.
- Whether training is needed first depends on the config's `checkpoint` value.
- A local config copy is reasonable, for example:
  - `configs/my_config_holobrain_gd_common.py`
- Using the original HF checkpoint in that local config is a valid way to export a pretrained model directly.

### Correct Commands
- Export, from `projects/holobrain`:
```bash
cd /home/sergey/Documents/projects/RoboOrchardLab/projects/holobrain
/home/sergey/Documents/projects/RoboOrchardLab/.venv/bin/python \
  export.py \
  --config configs/my_config_holobrain_gd_common.py \
  --workspace model_export
```
- Evaluate, from repo root:
```bash
cd /home/sergey/Documents/projects/RoboOrchardLab
export MODEL_TO_EVAL=/home/sergey/Documents/projects/RoboOrchardLab/projects/holobrain/model_export
export ROBOTWIN_DIR=/home/sergey/Documents/projects/RoboTwin_old/RoboTwin

PYTHONPATH=/home/sergey/Documents/projects/RoboOrchardLab:/home/sergey/Documents/projects/RoboOrchardLab/.venv/lib/python3.10/site-packages \
PATH="/home/sergey/miniconda3/envs/RoboTwin_old/bin:$PATH" \
/home/sergey/Documents/projects/RoboOrchardLab/.venv/bin/python \
projects/holobrain/holobrain_robotwin_eval/eval.py \
  --task_config my_config \
  --task_names beat_block_hammer,place_empty_cup \
  --model_config "$MODEL_TO_EVAL" \
  --model_prefix model/model \
  --robotwin_dir "$ROBOTWIN_DIR" \
  --num_workers 2 \
  --test_num 10 \
  --model_processor robotwin2_0_processor
```

### Runtime Notes
- Eval logs are written to `./eval_result/...` relative to repo root.
- If all tasks fail, outer `eval.py` still crashes at the end with:
  - `ZeroDivisionError` on `mean_success_rate = sum(...) / len(results)`
  - this is secondary; inspect the per-task `log.txt`
- One task uses one worker/GPU. Multiple GPUs are only used when multiple tasks are provided.
- RoboTwin progress messages such as `8 / 400` refer to per-episode step limit, not `test_num`.

## Follow-Up Investigation (March 13, 2026)
- We discussed offline evaluation of HoloBrain on Bridge-style tasks.
- Main conclusion:
  - SIMPLER looks like a better evaluation substrate than extending RoboTwin for this purpose, because it already provides Bridge/WidowX-style simulated tasks with success metrics.
- A concrete integration plan was written to:
  - `projects/holobrain/holobrain_robotwin_eval/HOLOBRAIN_SIMPLER_INTEGRATION_PLAN.md`
- Keep the note above brief here; use that document for architecture, risks, phases, and implementation details.

## Constant-Depth Fine-Tuning Note (March 17, 2026)
- We explored fine-tuning/eval with depth replaced by a constant value instead of disabling depth entirely.
- The constant-depth transform now lives in:
  - `robo_orchard_lab/dataset/robotwin/transforms.py`
  - class: `ReplaceDepthWithConstant`
- It supports a disable switch:
  - `value < 0` means "do not modify depth"
- `projects/holobrain/configs/config_robotwin_dataset.py` now references that transform in the RobotWin transform pipeline.
- Do not define this transform inside `projects/holobrain/configs/config_robotwin_dataset.py`:
  - doing so can break export with `RuntimeError: Set changed size during iteration` because the config module gets re-imported during processor build/load.
- The intended flow is:
  - train/fine-tune with the transform enabled
  - run `export.py`
  - evaluate using the exported processor, which should preserve the same depth replacement behavior
- We did not complete a final export verification in-session after the move; re-run export to confirm processor save/reload succeeds.

################### Human instructions ######################
# This line and the following text is human-created. It should not be modified, and should always stay at the end of AGENTS.md file.

## personal preferences
- be conservative about changing existing files:
  - always explain your changes
  - ask for permission before changing a file you (the agent)  have not previously modified
  - on the other hand, feel more freedom when modifying files created by you (the agent)
