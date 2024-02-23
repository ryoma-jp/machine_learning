FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN pip install notebook==7.1.0 torchinfo==1.8.0 ttach==0.0.3
