#! /bin/bash

# --- show usage ---
function usage {
    cat <<EOF
$(basename ${0}) is a tool for inference images using Raspberry Pi AI HAT

Usage:
    $(basename ${0}) [<options>]

Options:
    --model, -m           model to use (yolov8n, yolox_l_leaky, yolox_s_leaky, yolox_tiny, yolox_nano, deeplab_v3_mobilenet_v2, yolov8s_seg)
    --load-image, -l      load image (camera, image_file)
    --image-dir-path, -i  image directory path
    --version, -v         print version
    --help, -h            print this
EOF
}

# --- show version ---
function version {
    echo "$(basename ${0}) version 0.0.1 "
}

# --- argument processing ---
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

while [ $# -gt 0 ];
do
    case ${1} in

        --model|-m)
            MODEL=${2}
            shift
        ;;

        --load-image|-l)
            LOAD_IMAGE=${2}
            shift
        ;;

        --load-image|-l)
            LOAD_IMAGE=${2}
            shift
        ;;

        --image-dir-path|-i)
            IMAGE_DIR_PATH=${2}
            shift
        ;;
        
        --help|-h)
            usage
            exit 1
        ;;
        
        *)
            echo "[ERROR] Invalid option '${1}'"
            usage
            exit 1
        ;;
    esac
    shift
done

MODEL_DIR="$PWD/models/"
mkdir -p $MODEL_DIR

# see: https://github.com/hailo-ai/hailo_model_zoo
# supported models:
#   Object Detection
#       - yolov8n
#       - yolox_l_leaky
#       - yolox_s_leaky
#       - yolox_tiny
#       - yolox_nano # not working(nothing detected for unknown reason)
#   Semantic Segmentation
#       - deeplab_v3_mobilenet_v2
#   Instance Segmentation
#       - yolov8s_seg

MODEL_PATH=${MODEL_DIR}${MODEL}.hef
if [ -f "${MODEL_PATH}" ]; then
    echo "${MODEL_PATH} exists."
else
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8l/${MODEL}.hef -P $MODEL_DIR
fi

if [ ${LOAD_IMAGE} == "camera" ]; then
    echo "Load image from camera"
    PYTHON_SCRIPT="src/inference-camera-image.py"
    OPTIONS="--hef ${MODEL_PATH}"
elif [ ${LOAD_IMAGE} == "image_file" ]; then
    echo "Load image from image file"
    PYTHON_SCRIPT="src/inference-image-files.py"
    OPTIONS="--hef ${MODEL_PATH} --image_dir ${IMAGE_DIR_PATH}"
else
    echo "[ERROR] Invalid option '${LOAD_IMAGE}'"
    usage
    exit 1
fi

python3 ${PYTHON_SCRIPT} ${OPTIONS}
