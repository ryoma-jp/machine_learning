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
docker-compose build
docker-compose up -d
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

|Name|Task|
|---|---|
|[CIFAR-10 (PyTorch format)](https://www.cs.toronto.edu/~kriz/cifar.html)|Image Classification|
|[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)|Image Classification|

## Algorithms

### Image Classification

