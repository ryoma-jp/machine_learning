FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN pip install nvidia-pyindex onnx-graphsurgeon

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
