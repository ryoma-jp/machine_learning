# Benchmark

This directory contains the benchmarking performance of models.

## Results

| model                         | dataset   | task                 |   accuracy |   precision |   recall |         f1 |       AP50 |       AP75 | framework   |   framerate |
|:------------------------------|:----------|:---------------------|-----------:|------------:|---------:|-----------:|-----------:|-----------:|:------------|------------:|
| simple_cnn                    | cifar10   | image_classification |     0.8481 |    0.847851 |   0.8481 |   0.846878 | nan        | nan        | PyTorch     |  16113.5    |
| ssdlite320_mobilenet_v3_large | coco2014  | object_detection     |   nan      |  nan        | nan      | nan        |   0.286452 |   0.231867 | PyTorch     |     15.6111 |
| ssdlite320_mobilenet_v3_large | coco2017  | object_detection     |   nan      |  nan        | nan      | nan        |   0.240906 |   0.182672 | PyTorch     |     16.0749 |
| ssdlite320_mobilenet_v3_large | coco2014  | object_detection     |   nan      |  nan        | nan      | nan        |   0.286453 |   0.231882 | ONNX        |     19.4603 |
| ssdlite320_mobilenet_v3_large | coco2017  | object_detection     |   nan      |  nan        | nan      | nan        |   0.240918 |   0.182858 | ONNX        |     19.8145 |
| TVM-ACL_SimpleCNN_CIFAR10 | cifar10   | image_classification |     0.8482 |    0.847942 |   0.8482 | 0.84698 |        |        | TVM (Arm Compute Library) |     65.4214 |

- TVM (Arm Compute Library) models is runs on Raspberry Pi 4. 

## Environment

### PC

- **CPU**: Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz   2.90 GHz
- **GPU**: NVIDIA GeForce RTX 4070 Ti
    - **Driver Version**: 556.12
    - **CUDA Version**: 12.5 
- **RAM**: 64.0 GB
- **OS**: Windows 11 Home (23H2)
- **WSL2**: Ubuntu 22.04.4 LTS
    - **Docker**: version 27.0.3, build 7d4bcd8

### Raspberry Pi 4

- **Model**: Raspberry Pi 4 Model B Rev 1.4
    - `cat /proc/cpuinfo`
- **CPU**: ARM Cortex-A72 1.8GHz
    - `lscpu`
    - `vcgencmd get_config arm_freq`
- **RAM**: 8.0 GB
    - `free -tm`
- **OS**: Debian GNU/Linux 12 (bookworm)
    - `lsb_release -a`

#### Preparing

##### Install OS

see https://www.raspberrypi.com/software/

##### Install ARM NN

```
(Raspberry Pi)$ sudo apt install python3-pip
(Raspberry Pi)$ sudo pip3 install --break-system-packages virtualenv
(Raspberry Pi)$ cd <path/to/work>
(Raspberry Pi)$ wget https://github.com/ARM-software/armnn/releases/download/v24.08/ArmNN-linux-aarch64.tar.gz
(Raspberry Pi)$ mkdir ArmNN-linux-aarch64
(Raspberry Pi)$ tar -zxf ArmNN-linux-aarch64.tar.gz -C ArmNN-linux-aarch64
(Raspberry Pi)$ export LD_LIBRARY_PATH=$PWD/ArmNN-linux-aarch64
(Raspberry Pi)$ virtualenv -p python3 venv
(Raspberry Pi)$ source ./venv/bin/activate
(Raspberry Pi)(venv)$ pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime==2.14.0
(Raspberry Pi)(venv)$ pip3 install install numpy==1.26.4
(Raspberry Pi)(venv)$ mkdir test_data
(Raspberry Pi)(venv)$ wget -P test_data https://github.com/ARM-software/armnn/raw/branches/armnn_24_08/delegate/python/test/test_data/mock_model.tflite
(Raspberry Pi)(venv)$ python3 tutorial.py
```

# How to run benchmark

## Prepare

```bash
cd machene-learning
docker-compose build
docker-compose up -d
docker-compose exec ml bash
```

## Run

### PC

```bash
cd /workspace/benchmark
./run.sh
```
### Raspberry Pi 4

see [README.md for Arm Compute Library](../tvm/arm_compute_library/README.md)
