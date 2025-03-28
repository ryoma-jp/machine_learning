import os
from pathlib import Path
from urllib import request
from utils.utils import extract_tar, extract_zip
from data_loader.datasets import Cifar10ClassificationDataset, Coco2014ClassificationDataset, Coco2014Dataset, Coco2017Dataset

import torch
import torchvision
import torchvision.transforms as transforms

def collate_fn(batch):
    return tuple(zip(*batch))

class _DataLoaderCifar10PyTorch():
    """
    Data Loader for CIFAR-10 dataset for PyTorch
    This class provides to load CIFAR-10 dataset for PyTorch.
    """
    def __init__(self, 
                 resize=(32, 32), 
                 dataset_dir='/tmp/dataset', 
                 batch_size=32, 
                 shuffle_trainloader=True, 
                 shuffle_testloader=False, 
                 transform=None, 
                 load_trainset=True, 
                 load_testset=True) -> None:
        if (load_trainset):
            trainset = Cifar10ClassificationDataset(root=dataset_dir, train=True,
                                                    download=True, transform=transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=shuffle_trainloader, num_workers=2)
        else:
            self.trainloader = None
        if (load_testset):
            testset = Cifar10ClassificationDataset(root=dataset_dir, train=False,
                                                download=True, transform=transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=shuffle_testloader, num_workers=2)
        else:
            self.testloader = None
        self.class_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def inverse_normalize(self, img):
        transform = transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
        return transform(img)
    
class _DataLoaderCifar100PyTorch():
    """
    Data Loader for CIFAR-100 dataset for PyTorch
    This class provides to load CIFAR-100 dataset for PyTorch.
    """
    def __init__(self, 
                 resize=(32, 32), 
                 dataset_dir='/tmp/dataset', 
                 batch_size=32, 
                 shuffle_trainloader=True, 
                 shuffle_testloader=False, 
                 transform=None, 
                 load_trainset=True, 
                 load_testset=True) -> None:
        if (transform is None):
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transform
        
        if (load_trainset):
            trainset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True,
                                                    download=True, transform=transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=shuffle_trainloader, num_workers=2)
        else:
            self.trainloader = None
        if (load_testset):
            testset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False,
                                                download=True, transform=transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=shuffle_testloader, num_workers=2)
        else:
            self.testloader = None
        self.class_name = (
            'beaver', 'dolphin', 'otter', 'seal', 'whale',
            'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
            'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
            'bottles', 'bowls', 'cans', 'cups', 'plates',
            'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
            'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
            'bed', 'chair', 'couch', 'table', 'wardrobe',
            'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
            'bear', 'leopard', 'lion', 'tiger', 'wolf',
            'bridge', 'castle', 'house', 'road', 'skyscraper',
            'cloud', 'forest', 'mountain', 'plain', 'sea',
            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
            'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
            'crab', 'lobster', 'snail', 'spider', 'worm',
            'baby', 'boy', 'girl', 'man', 'woman',
            'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
            'maple', 'oak', 'palm', 'pine', 'willow',
            'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
            'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor',
        )
    
    def inverse_normalize(self, img):
        transform = transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
        return transform(img)
    
class _DataLoaderFood101PyTorch():
    """
    Data Loader for Food-101 dataset for PyTorch
    This class provides to load Food-101 dataset for PyTorch.
    """
    def __init__(self, 
                 resize=(128, 128), 
                 dataset_dir='/tmp/dataset', 
                 batch_size=32, 
                 shuffle_trainloader=True, 
                 shuffle_testloader=False, 
                 transform=None, 
                 load_trainset=True, 
                 load_testset=True) -> None:
        if (transform is None):
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transform
        
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
        
        if (load_trainset):
            train_images = torchvision.datasets.ImageFolder(root=Path(dataset_dir, 'food-101/train_images/'), transform=transform)
            self.train_file_list = [train_file[len(f'{dataset_dir}/food-101/train_images/'):] for train_file, _ in train_images.imgs]
            self.trainloader = torch.utils.data.DataLoader(train_images, batch_size=batch_size, shuffle=shuffle_trainloader, num_workers=2)
        else:
            self.trainloader = None
        if (load_testset):
            test_images = torchvision.datasets.ImageFolder(root=Path(dataset_dir, 'food-101/test_images/'), transform=transform)
            self.test_file_list = [test_file[len(f'{dataset_dir}/food-101/test_images/'):] for test_file, _ in test_images.imgs]
            self.testloader = torch.utils.data.DataLoader(test_images, batch_size=batch_size, shuffle=shuffle_testloader, num_workers=2)
        else:
            self.testloader = None
        self.class_name = train_images.classes
    
    def inverse_normalize(self, img):
        transform = transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
        return transform(img)
    
class _DataLoaderOfficeHomePyTorch():
    """
    Data Loader for OfficeHome dataset for PyTorch
    This class provides to load OfficeHome dataset for PyTorch.
    """
    def __init__(self, 
                 resize=(227, 227), 
                 dataset_dir='/tmp/dataset', 
                 batch_size=32, 
                 shuffle_trainloader=True, 
                 shuffle_testloader=False, 
                 transform=None, 
                 load_trainset=True, 
                 load_testset=True) -> None:
        # --- Extract dataset ---
        filename = 'OfficeHomeDataset_10072016.zip'
        filepath = Path(dataset_dir, filename)
        extract_zip(str(filepath), dataset_dir)
        
        # --- Load dataset ---
        if (transform is None):
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transform
        
        if (load_trainset):
            train_images = torchvision.datasets.ImageFolder(root=Path(dataset_dir, 'OfficeHomeDataset_10072016/Art/'), transform=transform)
            self.train_file_list = [train_file[len(f'{dataset_dir}/OfficeHomeDataset_10072016/Art/'):] for train_file, _ in train_images.imgs]
            self.trainloader = torch.utils.data.DataLoader(train_images, batch_size=batch_size, shuffle=shuffle_trainloader, num_workers=2)
        else:
            self.trainloader = None
        if (load_testset):
            test_images = torchvision.datasets.ImageFolder(root=Path(dataset_dir, 'OfficeHomeDataset_10072016/Art/'), transform=transform)
            self.test_file_list = [test_file[len(f'{dataset_dir}/OfficeHomeDataset_10072016/Art/'):] for test_file, _ in test_images.imgs]
            self.testloader = torch.utils.data.DataLoader(test_images, batch_size=batch_size, shuffle=shuffle_testloader, num_workers=2)
        else:
            self.testloader = None
        self.class_name = train_images.classes
        
class _DataLoaderCoco2014PyTorch():
    """
    Data Loader for COCO2014 dataset for PyTorch
    """
    def __init__(self, 
                 resize=(224, 224), 
                 dataset_dir='/tmp/dataset', 
                 batch_size=32, 
                 shuffle_trainloader=True, 
                 shuffle_testloader=False, 
                 transform=None, 
                 load_trainset=True, 
                 load_testset=True) -> None:
        # --- Define transform ---
        #   - ToTensor: Convert PIL Image to Tensor
        #       - Convert shape(HWC -> CHW) and range([0, 255] -> [0.0, 1.0])
        #       - https://github.com/pytorch/vision/blob/fbb4cc54ed521ba912f50f180dc16a213775bf5c/torchvision/transforms/transforms.py#L107
        #   - Normalize: Normalize the image with mean and standard deviation
        if (transform is None):
            transform = transforms.Compose([
                transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            ])
        else:
            transform = transform
        
        if (load_trainset):
            trainset = Coco2014Dataset(root=dataset_dir, train=True,
                                                    input_size=resize,
                                                    download=True, transform=transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=shuffle_trainloader,
                                                    collate_fn=collate_fn, num_workers=8)
            self.class_name = trainset.df_annotations['category_name'].unique().tolist()
        else:
            self.trainloader = None
        if (load_testset):
            testset = Coco2014Dataset(root=dataset_dir, train=False,
                                                input_size=resize,
                                                download=True, transform=transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=shuffle_testloader,
                                                    collate_fn=collate_fn, num_workers=8)
            if (self.class_name is None):
                self.class_name = testset.df_annotations['category_name'].unique().tolist()
        else:
            self.testloader = None
    
    def inverse_normalize(self, img):
        return img * 255.0

class _DataLoaderCoco2014ClassificationPyTorch():
    """
    Data Loader for COCO2014 classification dataset for PyTorch
    This class provides to load COCO2014 classification dataset for PyTorch.
    """
    def __init__(self, 
                 resize=(224, 224), 
                 dataset_dir='/tmp/dataset', 
                 batch_size=32, 
                 shuffle_trainloader=True, 
                 shuffle_testloader=False, 
                 transform=None, 
                 load_trainset=True, 
                 load_testset=True) -> None:
        if (transform is None):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transform
        
        if (load_trainset):
            trainset = Coco2014ClassificationDataset(root=dataset_dir, train=True,
                                                    download=True, transform=transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=shuffle_trainloader, num_workers=8)
            self.class_name = trainset.df_dataset['category_name'].unique().tolist()
        else:
            self.trainloader = None
        if (load_testset):
            testset = Coco2014ClassificationDataset(root=dataset_dir, train=False,
                                                download=True, transform=transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=shuffle_testloader, num_workers=8)
            if (self.class_name is None):
                self.class_name = testset.df_dataset['category_name'].unique().tolist()
        else:
            self.testloader = None
    
    def inverse_normalize(self, img):
        transform = transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
        return transform(img)

class _DataLoaderCoco2017PyTorch():
    """
    Data Loader for COCO2017 dataset for PyTorch
    """
    def __init__(self, 
                 resize=(224, 224), 
                 dataset_dir='/tmp/dataset', 
                 batch_size=32, 
                 shuffle_trainloader=True, 
                 shuffle_testloader=False, 
                 transform=None, 
                 load_trainset=True, 
                 load_testset=True) -> None:
        # --- Define transform ---
        #   - ToTensor: Convert PIL Image to Tensor
        #       - Convert shape(HWC -> CHW) and range([0, 255] -> [0.0, 1.0])
        #       - https://github.com/pytorch/vision/blob/fbb4cc54ed521ba912f50f180dc16a213775bf5c/torchvision/transforms/transforms.py#L107
        #   - Normalize: Normalize the image with mean and standard deviation
        if (transform is None):
            transform = transforms.Compose([
                transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            ])
        else:
            transform = transform
        
        self.class_name = None
        if (load_trainset):
            trainset = Coco2017Dataset(root=dataset_dir, train=True,
                                                    input_size=resize,
                                                    download=True, transform=transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=shuffle_trainloader,
                                                    collate_fn=collate_fn, num_workers=8)
            self.class_name = trainset.df_annotations['category_name'].unique().tolist()
        else:
            self.trainloader = None

        if (load_testset):
            testset = Coco2017Dataset(root=dataset_dir, train=False,
                                                input_size=resize,
                                                download=True, transform=transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=shuffle_testloader,
                                                    collate_fn=collate_fn, num_workers=8)
            if (self.class_name is None):
                self.class_name = testset.df_annotations['category_name'].unique().tolist()
        else:
            self.testloader = None

    
    def inverse_normalize(self, img):
#        transform = transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
#        return transform(img) * 255.0
        return img * 255.0

class DataLoader():
    """
    Data Loader for PyTorch
    This class provides to load datasets for PyTorch.
    
    Attributes:
    DATASET_NAMES: List of dataset names.
    DEFAULT_SIZE: Dictionary of default size for each dataset.
    FUNCTION_TABLE: Dictionary of functions to load each dataset.
    
    Methods:
    __init__(dataset_name, resize, dataset_dir, batch_size): Initialize the class.
    """
    DATASET_NAMES = ['cifar10_pytorch']
    DEFAULT_SIZE = {
        'cifar10_pytorch': (32, 32),
        'cifar100_pytorch': (32, 32),
        'food101_pytorch': (128, 128),
        'officehome_pytorch': (227, 227),
        'coco2014_classification_pytorch': (224, 224),
        'coco2014_pytorch': (224, 224),
        'coco2017_pytorch': (224, 224),
    }
    FUNCTION_TABLE = {
        'cifar10_pytorch': _DataLoaderCifar10PyTorch,
        'cifar100_pytorch': _DataLoaderCifar100PyTorch,
        'food101_pytorch': _DataLoaderFood101PyTorch,
        'officehome_pytorch': _DataLoaderOfficeHomePyTorch,
        'coco2014_classification_pytorch': _DataLoaderCoco2014ClassificationPyTorch,
        'coco2014_pytorch': _DataLoaderCoco2014PyTorch,
        'coco2017_pytorch': _DataLoaderCoco2017PyTorch,
    }
    
    def __init__(self, 
                 dataset_name=DATASET_NAMES[0], 
                 resize=None, 
                 dataset_dir='/tmp/dataset', 
                 batch_size=32, 
                 transform=None,
                 load_trainset=True,
                 load_testset=True) -> None:
        if (resize is None):
            resize = self.DEFAULT_SIZE[dataset_name]
            
        self.dataset = self.FUNCTION_TABLE[dataset_name](
            resize=resize,
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            transform=transform,
            load_trainset=load_trainset,
            load_testset=load_testset)

