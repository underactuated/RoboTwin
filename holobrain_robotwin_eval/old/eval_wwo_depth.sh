# UPDATE: it does not seem possible to set with_depth=False in my_config_holobrain_gd_common.py before exporting model. It causes input mismatch error in the checkpoint. Does it mean a model should be pretrained with with parameter, since it changes model input sizes?

# This shell will be running two evaluations (with and without depth) of a subset of robotwin tasks 'tasks', testing each task for 'test_num' times.

test_num=2

robot_config_source=my_config_holobrain_gd_common.py

eval_tasks (tasks, test_num, depth_flag, out_file) {
    		

}


cd /home/sergey/Documents/projects/RoboOrchardLab
export MODEL_TO_EVAL=/home/sergey/Documents/projects/RoboOrchardLab/projects/holobrain/model_export
export ROBOTWIN_DIR=/home/sergey/Documents/projects/RoboTwin_old/RoboTwin

eval_tasks(tasks, test_num, true, "eval_out_w_depth.txt")
eval_tasks(tasks, test_num, false, "eval_out_wo_depth.txt")
