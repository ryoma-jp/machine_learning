# Benchmark

This directory contains the benchmarking performance of models.

## Results

```
                           model   dataset              task framework      AP50      AP75  framerate
0  ssdlite320_mobilenet_v3_large  coco2014  object_detection   PyTorch  0.286452  0.231867  16.345513
1  ssdlite320_mobilenet_v3_large  coco2017  object_detection   PyTorch  0.240906  0.182672  16.587228
2  ssdlite320_mobilenet_v3_large  coco2014  object_detection      ONNX  0.286453  0.231882  21.830256
3  ssdlite320_mobilenet_v3_large  coco2017  object_detection      ONNX  0.240918  0.182858  21.273549
```

## Environment

- **CPU**: Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz   2.90 GHz
- **GPU**: NVIDIA GeForce RTX 4070 Ti
    - **Driver Version**: 556.12
    - **CUDA Version**: 12.5 
- **RAM**: 64.0 GB
- **OS**: Windows 11 Home (23H2)
- **WSL2**: Ubuntu 22.04.4 LTS
    - **Docker**: version 27.0.3, build 7d4bcd8

# How to run benchmark

## Prepare

```bash
cd machene-learning
docker-compose build
docker-compose up -d
docker-compose exec ml bash
```

## Run

```bash
cd /workspace/benchmark
./run.sh
```
