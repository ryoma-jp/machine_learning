#! /bin/bash

MODEL_DIR="$PWD/models/"
mkdir -p $MODEL_DIR

# see: https://github.com/hailo-ai/hailo_model_zoo

# Object Detection
#MODEL="yolov8n.hef"
#MODEL="yolox_l_leaky.hef"
#MODEL="yolox_s_leaky.hef"
MODEL="yolox_tiny.hef"
#MODEL="yolox_nano.hef" # not working(nothing detected for unknown reason)

# Semantic Segmentation
#MODEL="deeplab_v3_mobilenet_v2.hef"

# Instance Segmentation
#MODEL="yolov8s_seg.hef"

if [ -f $MODEL_DIR$MODEL ]; then
    echo "$MODEL_DIR$MODEL exists."
else
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8l/${MODEL} -P $MODEL_DIR
fi

python3 src/inference-camera-image.py --hef $MODEL_DIR$MODEL
