
import os
from pathlib import Path
from urllib import request
from utils.utils import extract_tar

import torch
import torchvision
import torchvision.transforms as transforms

class _DataLoaderCifar10PyTorch():
    '''Data Loader for CIFAR-10 dataset for PyTorch
    This class provides to load CIFAR-10 dataset for PyTorch.
    '''
    def __init__(self, dataset_dir='/tmp/dataset', batch_size=32) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                            download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

        self.classe_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
class _DataLoaderFood101PyTorch():
    '''Data Loader for ObjectNet dataset for PyTorch
    This class provides to load ObjectNet dataset for PyTorch.
    '''
    def __init__(self, dataset_dir='/tmp/dataset', batch_size=32) -> None:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        url = ' http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
        filename = 'food-101.tar.gz'
        filepath = Path(dataset_dir, filename)
        if (not filepath.exists()):
            # --- Download and extract dataset ---
            Path(dataset_dir).mkdir(exist_ok=True, parents=True)
            print(f'Download {filename} from {url}')
            request.urlretrieve(url, filepath)
            print(f'Extract {filename} to {dataset_dir}')
            extract_tar(filepath, dataset_dir)
            
            # --- Load meta data ---
            with open(f'{dataset_dir}/food-101/meta/train.txt', 'r') as f:
                train_file_list = [line + '.jpg' for line in f.read().splitlines()]
            with open(f'{dataset_dir}/food-101/meta/test.txt', 'r') as f:
                test_file_list = [line + '.jpg' for line in f.read().splitlines()]
            with open(f'{dataset_dir}/food-101/meta/classes.txt', 'r') as f:
                class_list = [line for line in f.read().splitlines()]
            
            # --- Create symbolic links ---
            train_images_dir = Path(dataset_dir, 'food-101/train_images')
            for class_ in class_list:
                os.makedirs(Path(train_images_dir, class_), exist_ok=True)
            for train_file in train_file_list:
                os.symlink(Path(dataset_dir, 'food-101/images', train_file), Path(train_images_dir, train_file))
            
            test_images_dir = Path(dataset_dir, 'food-101/test_images')
            for class_ in class_list:
                os.makedirs(Path(test_images_dir, class_), exist_ok=True)
            for test_file in test_file_list:
                os.symlink(Path(dataset_dir, 'food-101/images', test_file), Path(test_images_dir, test_file))
        
        train_images = torchvision.datasets.ImageFolder(root=Path(dataset_dir, 'food-101/train_images/'), transform=transform)
        self.trainloader = torch.utils.data.DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=2)
        
        test_images = torchvision.datasets.ImageFolder(root=Path(dataset_dir, 'food-101/test_images/'), transform=transform)
        self.testloader = torch.utils.data.DataLoader(test_images, batch_size=batch_size, shuffle=True, num_workers=2)
        
        self.class_name = train_images.classes


class DataLoader():
    '''Data Loader
    Base class for data loaders. All data loaders should inherit from this class.
    This class privides to load below datasets:
        - cifar10_pytorch (CIFAR-10 dataset for PyTorch)
    '''
    DATASET_NAMES = ['cifar10_pytorch']
    FUNCTION_TABLE = {
        'cifar10_pytorch': _DataLoaderCifar10PyTorch,
        'food101_pytorch': _DataLoaderFood101PyTorch,
    }
    
    def __init__(self, dataset_name=DATASET_NAMES[0], dataset_dir='/tmp/dataset', batch_size=32) -> None:
        self.dataset = self.FUNCTION_TABLE[dataset_name](dataset_dir, batch_size)
        
