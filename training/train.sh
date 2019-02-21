# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH="{path_to_config}/ssd_mobilenet_v2_coco.config" #you can use any config file if you edit it to be appropriate to the hand data
MODEL_DIR="{Where to save the model}"
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr