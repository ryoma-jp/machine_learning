# Machine Leanring

This repository is implemented machine learning program samples.

## Environment

Docker compose is used.  
Please modify the parameters in `.env` file.

|parameter|description|
|---|---|
|port|TCP/IP port of host PC. Default is 35000.|

### Build Docker Image and Run Docker Container

```bash
$ docker-compose build
$ docker-compose up -d
$ docker-compose exec ml bash
```

### Jupyter Notebook

Access to `http://localhost:35000` with browser.  
You can check the token by running below command.

```bash
docker-compose logs ml
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

