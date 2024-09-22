# Inference on ARM CPU

Copy these files and the converted model to the target device, and run inference.

## Prepare Model for Arm Compute Library

Run `notebooks/001_ImageClassification-CIFAR10-SimpleCNN-PyTorch.ipynb`.  
The model will be saved as `001_ImageClassification-CIFAR10-SimpleCNN-PyTorch/model/arm_compute_library/model.so`.

## Virtual Environment
```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Run
```bash
$ ls
README.md  acl_inference.py  arm_compute_library  requirements.txt  venv
$ python3 acl_inference.py
```