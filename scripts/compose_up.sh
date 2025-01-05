#! /bin/bash

SERVICES="ml benchmark yolox"

HAILO_WHL1="compiler/hailo/docker/whl/hailo_dataflow_compiler-3.29.0-py3-none-linux_x86_64.whl"
HAILO_WHL2="compiler/hailo/docker/whl/hailo_model_zoo-2.13.0-py3-none-any.whl"
HAILO_WHL3="compiler/hailo/docker/whl/hailort-4.19.0-cp310-cp310-linux_x86_64.whl"

if [ -f $HAILO_WHL1 ] && [ -f $HAILO_WHL2 ] && [ -f $HAILO_WHL3 ]; then
    SERVICES="$SERVICES hailo_compiler"
fi

docker compose up --build -d $SERVICES
