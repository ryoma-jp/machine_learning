#! /bin/bash

# --- show usage ---
function usage {
    cat <<EOF
$(basename ${0}) is a tool for starting docker compose services

Usage:
    $(basename ${0}) [<options>]

Options:
    --device, -d      device to use (PC, RaspberryPi)
    --version, -v     print version
    --help, -h        print this
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

        --device|-d)
            DEVICE=${2}
            shift
        ;;

        --version|-v)
            version
            exit 1
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

SUPPORT_DEVICES=("PC" "RaspberryPi")
if [[ ! " ${SUPPORT_DEVICES[@]} " =~ " ${DEVICE} " ]]; then
    echo "[ERROR] Unsupported device '${DEVICE}'"
    usage
    exit 1
fi

if [ "${DEVICE}" == "PC" ]; then
    SERVICES="ml benchmark yolox"

    HAILO_WHL1="compiler/hailo/docker/whl/hailo_dataflow_compiler-3.29.0-py3-none-linux_x86_64.whl"
    HAILO_WHL2="compiler/hailo/docker/whl/hailo_model_zoo-2.13.0-py3-none-any.whl"
    HAILO_WHL3="compiler/hailo/docker/whl/hailort-4.19.0-cp310-cp310-linux_x86_64.whl"

    if [ -f $HAILO_WHL1 ] && [ -f $HAILO_WHL2 ] && [ -f $HAILO_WHL3 ]; then
        SERVICES="$SERVICES hailo_compiler"
    fi
elif [ "${DEVICE}" == "RaspberryPi" ]; then
    SERVICES="rpi_ai_hat"
fi

docker compose up --build -d $SERVICES
