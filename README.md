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

|Name|Task|URL|
|---|---|---|
|CIFAR-10 (PyTorch format)|Image Classification|[LINK](https://www.cs.toronto.edu/~kriz/cifar.html)|


## Algorithms

### Image Classification

