# Benchmark

This directory contains the benchmarking performance of models.

```
                           model   dataset              task framework      AP50      AP75 framerate
0  ssdlite320_mobilenet_v3_large  coco2014  object_detection   PyTorch  0.286452  0.231867     T.B.D
1  ssdlite320_mobilenet_v3_large  coco2017  object_detection   PyTorch  0.240906  0.182672     T.B.D
2  ssdlite320_mobilenet_v3_large  coco2014  object_detection      ONNX  0.286453  0.231882     T.B.D
3  ssdlite320_mobilenet_v3_large  coco2017  object_detection      ONNX  0.240918  0.182858     T.B.D
```

# How to run benchmark

```bash
cd /workspace/benchmark
./run.sh
```
