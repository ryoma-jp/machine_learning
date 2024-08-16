# Benchmark

This directory contains the benchmarking performance of models.

```
                           model   dataset              task framework      AP50      AP75  framerate
0  ssdlite320_mobilenet_v3_large  coco2014  object_detection   pytorch  0.286452  0.231867          0
1  ssdlite320_mobilenet_v3_large  coco2017  object_detection   pytorch  0.240906  0.182672          0
```

# How to run benchmark

```bash
cd /workspace/benchmark
./run.sh
```
