#! /bin/bash

cat <<EOF > .env
UID=$(id -u)
GID=$(id -g)
UNAME=ml_dev
ML_PORT=35000
EOF
