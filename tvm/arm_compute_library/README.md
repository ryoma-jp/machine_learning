# Inference on ARM CPU

Copy these files, the converted model and test data to the target device, and run inference.

## Prepare Model for Arm Compute Library

Run `notebooks/001_ImageClassification-CIFAR10-SimpleCNN-PyTorch.ipynb`.  
The model will be saved as `001_ImageClassification-CIFAR10-SimpleCNN-PyTorch/model/arm_compute_library/model.so`.

## Prepare Test Data

Run `run_simple-cnn-cifar10.sh` in `benchmark` directory.  
The test data will be saved as `benchmark/outputs/PyTorch_SimpleCNN_CIFAR10/input_tensors`.

```
# cd benchmark
# ./run_simple-cnn-cifar10.sh
# scp outputs/PyTorch_SimpleCNN_CIFAR10/input_tensors <user>@<target>:/path/to/acl_inference
```

## Virtual Environment
```
$ cd /path/to/acl_inference
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Run
```
$ ls
README.md  acl_inference.py  arm_compute_library  requirements.txt  venv
$ python3 acl_inference.py
```