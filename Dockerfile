FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN apt update && \
    apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu fonts-liberation

RUN pip install nvidia-pyindex==1.0.9 onnx-graphsurgeon==0.5.5

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
