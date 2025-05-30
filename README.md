# Machine Leanring

This repository is implemented machine learning program samples.

## Environment

Docker compose is used.  
Please modify the parameters in `make_env.sh` file.

|parameter|description|
|---|---|
|UID|User ID of host PC. Default is the ID of the login user.|
|GID|Group ID of host PC. Default is the ID of the login user.|
|UNAME|User name of host PC. Default is "ml_dev".|
|ML_PORT|TCP/IP port of host PC. Default is 35000.|

### Build Docker Image and Run Docker Container

```bash
cd /path/to/machine-learning
./scripts/make_env.sh
```

#### On PC

```bash
./scripts/compose_up.sh -d PC
```

#### On Raspberry Pi

```bash 
./scripts/compose_up.sh -d RaspberryPi
```

#### Login to the development environment

```bash
docker compose exec ml bash
```

#### Login to the benchmark environment

```bash
docker compose exec benchmark bash
```

#### Login to the Raspberry Pi AI HAT+ environment

```bash
docker compose exec rpi_ai_hat bash
```

### Jupyter Notebook

Access to `http://localhost:35000` with browser.  
You can check the token by running below command.

```bash
docker compose logs ml
```

### Training YOLO

```bash
docker compose exec yolox bash
cd external/yolox
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
mkdir weights
```

#### YOLOX-Tiny

```bash
wget -P ./weights https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_tiny.pth
python -m yolox.tools.train -f exps/default/yolox_tiny.py -c ./weights/yolox_tiny.pth -d 1 -b 8 --fp16 -o
python tools/export_onnx.py --output-name weights/yolox_tiny.onnx -n yolox-tiny -c weights/yolox_tiny.pth
```

#### YOLOX-Nano

```bash
wget -P ./weights https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_nano.pth
python -m yolox.tools.train -f exps/default/yolox_nano.py -c ./weights/yolox_nano.pth -d 1 -b 8 --fp16 -o
python tools/export_onnx.py --output-name weights/yolox_nano.onnx -n yolox-nano -c weights/yolox_nano.pth
```

### Evaluation YOLO

#### YOLOX-Tiny

```bash
python -m yolox.tools.eval -f exps/default/yolox_tiny.py -c ./weights/yolox_tiny.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
```

#### YOLOX-Nano

```bash
python -m yolox.tools.eval -f exps/default/yolox_nano.py -c ./weights/yolox_nano.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
```

### Compile YOLOX for Hailo8

#### YOLOX-Tiny

```bash
docker compose exec hailo_compiler bash
cd compiler/hailo
ln -s /workspace/external/yolox/YOLOX/YOLOX_outputs/yolox_tiny/yolox_tiny.onnx weights/yolox_tiny.onnx
hailomz compile --ckpt weights/yolox_tiny.onnx --calib-path /dataset/coco2017/val2017/ --yaml cfg/networks/yolox_tiny.yaml --model-script cfg/alls/yolox_tiny.alls
```

#### YOLOX-Nano

notes: 
YOLOX-Nano is not working on Hailo8.  
Nothing is detected for unknown reasons.

```bash
docker compose exec hailo_compiler bash
cd compiler/hailo
ln -s /workspace/external/yolox/YOLOX/YOLOX_outputs/yolox_nano/yolox_nano.onnx weights/yolox_nano.onnx
hailomz compile --ckpt weights/yolox_nano.onnx --calib-path /dataset/coco2017/val2017/ --yaml cfg/networks/yolox_nano.yaml --model-script cfg/alls/yolox_nano.alls
```

## Dataset

Open Dataset is used.  
Supported dataset is below.

|Name|Task|License / Term of Use|
|---|---|---|
|[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)|Image Classification|Unknown (Not specified)|
|[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)|Image Classification|Unknown (Not specified)|
|[Office-Home Dataset](https://www.hemanthdv.org/officeHomeDataset.html)|Domain Adaptaion|Custom (non-commercial research and educational purposes). See [Fair Use Notice](https://www.hemanthdv.org/officeHomeDataset.html)|
|[The Human Activity Recognition Trondheim](https://archive.ics.uci.edu/dataset/779/harth)|Activity Classification|CC BY 4.0|

### References

- [Datasets (Image Classification)](https://paperswithcode.com/datasets?task=image-classification)

## Algorithms

### Image Classification

|Name|Framework|Description|
|---|---|---|
|[SimpleCNN](./models/pytorch/simple_cnn.py)|PyTorch|Simple structure base on convolution layers|
|[VGG16](./models/pytorch/vgg16.py)|PyTorch|VGG16|

### Explainable

Explainable methods to show the reason where the model looked when inference.

|Method|Description|Reference|
|---|---|---|
|[Grad-CAM](./explainable_ai/pytorch/grad_cam.py)|Weight the 2D activations by the average gradient|[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam/tree/51ae19245f655cf0ee334db2a945ceb1a4d6df59)|
|[Eigen-CAM](./explainable_ai/pytorch/eigen_cam.py)|Takes the first principle component of the 2D Activations (no class discrimination, but seems to give great results)|[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam/tree/51ae19245f655cf0ee334db2a945ceb1a4d6df59)|
|[Parameter Space Saliency Maps](./explainable_ai/pytorch/pss.py)| Identify and analyze the network parameters,  which are responsible for erroneous decisions.|[parameter-space-saliency](https://github.com/LevinRoman/parameter-space-saliency/tree/0e3b3d69c6e222aee6af0264d7ce3ddc6d19744e)|

### Domain Adaptation

### The Human Activity Recognition Trondheim (T.B.D)

|Name|Framework|Description|
|---|---|---|
|[LightGBM](./models/lightgbm/lgbm_classification.py)|LightGBM|The Human Activity Recognition Trondheim|

### Build TVM

#### For ARM Compute Library

T.B.D

```
https://tvm.apache.org/docs/how_to/deploy/arm_compute_lib.html
https://tvm.apache.org/docs/install/from_source.html
https://tvm.apache.org/docs/how_to/tutorials/cross_compilation_and_rpc.html

export LD_LIBRARY_PATH=/home/ryoichi/work/github/machine_learning/external/lib/arm_compute-v24.09-linux-aarch64-cpu-bin/lib/:$LD_LIBRARY_PATH

export PATH=/home/ryoichi/work/github/machine_learning/external/lib/arm_compute-v24.09-linux-aarch64-cpu-bin/bin/:$PATH

LD_LIBRARY_PATH=/home/ryoichi/work/github/machine_learning/external/lib/arm_compute-v24.09-linux-aarch64-cpu-bin/lib/:$LD_LIBRARY_PATH cmake --build . --parallel 4
```

#### For Raspberry Pi AI HAT+

see [Raspberry Pi AI HAT+](./benchmark/rpi_ai_hat/README.md)

