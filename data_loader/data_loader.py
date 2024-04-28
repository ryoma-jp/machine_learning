
import os
from pathlib import Path
from urllib import request
from utils.utils import extract_tar, extract_zip
from data_loader.datasets import Coco2014ClassificationDataset

import torch
import torchvision
import torchvision.transforms as transforms

class _DataLoaderCifar10PyTorch():
    '''Data Loader for CIFAR-10 dataset for PyTorch
    This class provides to load CIFAR-10 dataset for PyTorch.
    '''
    def __init__(self, resize=(32, 32), dataset_dir='/tmp/dataset', batch_size=32, shuffle_trainloader=True, shuffle_testloader=False) -> None:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=shuffle_trainloader, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                            download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=shuffle_testloader, num_workers=2)

        self.classe_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def inverse_normalize(self, img):
        transform = transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
        return transform(img)
    
class _DataLoaderFood101PyTorch():
    '''Data Loader for ObjectNet dataset for PyTorch
    This class provides to load ObjectNet dataset for PyTorch.
    '''
    def __init__(self, resize=(128, 128), dataset_dir='/tmp/dataset', batch_size=32, shuffle_trainloader=True, shuffle_testloader=False) -> None:
        transform = transforms.Compose([
            transforms.Resize(resize),
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
        self.train_file_list = [train_file[len(f'{dataset_dir}/food-101/train_images/'):] for train_file, _ in train_images.imgs]
        self.trainloader = torch.utils.data.DataLoader(train_images, batch_size=batch_size, shuffle=shuffle_trainloader, num_workers=2)
        
        test_images = torchvision.datasets.ImageFolder(root=Path(dataset_dir, 'food-101/test_images/'), transform=transform)
        self.test_file_list = [test_file[len(f'{dataset_dir}/food-101/test_images/'):] for test_file, _ in test_images.imgs]
        self.testloader = torch.utils.data.DataLoader(test_images, batch_size=batch_size, shuffle=shuffle_testloader, num_workers=2)
        
        self.class_name = train_images.classes
    
    def inverse_normalize(self, img):
        transform = transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
        return transform(img)
    
class _DataLoaderOfficeHomePyTorch():
    '''Data Loader for ObjectNet dataset for PyTorch
    This class provides to load ObjectNet dataset for PyTorch.
    '''
    def __init__(self, resize=(227, 227), dataset_dir='/tmp/dataset', batch_size=32, shuffle_trainloader=True, shuffle_testloader=False) -> None:
        # --- Extract dataset ---
        filename = 'OfficeHomeDataset_10072016.zip'
        filepath = Path(dataset_dir, filename)
        extract_zip(str(filepath), dataset_dir)
        
        # --- Load dataset ---
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_images = torchvision.datasets.ImageFolder(root=Path(dataset_dir, 'OfficeHomeDataset_10072016/Art/'), transform=transform)
        self.train_file_list = [train_file[len(f'{dataset_dir}/OfficeHomeDataset_10072016/Art/'):] for train_file, _ in train_images.imgs]
        self.trainloader = torch.utils.data.DataLoader(train_images, batch_size=batch_size, shuffle=shuffle_trainloader, num_workers=2)
        
        self.class_name = train_images.classes
        
class _DataLoaderCoco2014ClassificationPyTorch():
    '''Data Loader for COCO2014 Classification dataset for PyTorch
    This class provides to load COCO2014 classification dataset modified from the original for PyTorch.
    '''
    def __init__(self, resize=(224, 224), dataset_dir='/tmp/dataset', batch_size=32, shuffle_trainloader=True, shuffle_testloader=False) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = Coco2014ClassificationDataset(root=dataset_dir, train=True,
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=shuffle_trainloader, num_workers=2)

#        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
#                                            download=True, transform=transform)
#        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                                shuffle=shuffle_testloader, num_workers=2)
#
#        self.classe_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def inverse_normalize(self, img):
        transform = transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
        return transform(img)

class DataLoader():
    '''Data Loader
    Base class for data loaders. All data loaders should inherit from this class.
    This class privides to load below datasets:
        - cifar10_pytorch (CIFAR-10 dataset for PyTorch)
        - food101_pytorch (Food-101 dataset for PyTorch)
        - coco2014_classification_pytorch (COCO2014 classification dataset(modified from original) for PyTorch)
    '''
    DATASET_NAMES = ['cifar10_pytorch']
    DEFAULT_SIZE = {
        'cifar10_pytorch': (32, 32),
        'food101_pytorch': (128, 128),
        'officehome_pytorch': (227, 227),
        'coco2014_classification_pytorch': (224, 224),
    }
    FUNCTION_TABLE = {
        'cifar10_pytorch': _DataLoaderCifar10PyTorch,
        'food101_pytorch': _DataLoaderFood101PyTorch,
        'officehome_pytorch': _DataLoaderOfficeHomePyTorch,
        'coco2014_classification_pytorch': _DataLoaderCoco2014ClassificationPyTorch,
    }
    
    def __init__(self, dataset_name=DATASET_NAMES[0], resize=None, dataset_dir='/tmp/dataset', batch_size=32) -> None:
        if (resize is None):
            resize = self.DEFAULT_SIZE[dataset_name]
            
        self.dataset = self.FUNCTION_TABLE[dataset_name](resize=resize, dataset_dir=dataset_dir, batch_size=batch_size)
        
