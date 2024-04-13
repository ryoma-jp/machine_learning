FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN pip install nvidia-pyindex onnx-graphsurgeon
RUN pip install notebook==7.1.0 torchinfo==1.8.0 ttach==0.0.3 tensorflow==2.15.0 onnx==1.15.0 sng4onnx==1.0.1 onnx2tf==1.19.12
