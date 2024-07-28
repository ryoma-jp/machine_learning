FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN pip install nvidia-pyindex onnx-graphsurgeon
RUN pip install -r requirements.txt
