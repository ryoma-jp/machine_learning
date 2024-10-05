FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN apt update && \
    apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

RUN pip install nvidia-pyindex onnx-graphsurgeon

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
