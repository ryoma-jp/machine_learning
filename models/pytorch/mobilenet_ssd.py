
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
from torchinfo import summary
from tqdm import tqdm
from models.pytorch.pytorch_model_base import PyTorchModelBase
from .mobilenet_ssd_models.mobilenetv2_ssd_lite import create_mobilenetv2_ssd_lite
from .config import mobilenetv1_ssd_config as config
from .ssd.data_preprocessing import TrainAugmentation, TestTransform
from .ssd.ssd import MatchPrior

# --- for Test ---
import logging
import os
import cv2
import pathlib
class VOCDataset:
    
    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list
            
            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')


        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

class MobileNetSSD(PyTorchModelBase):
    def __init__(self, device, version, input_size, num_classes, output_dir='outputs', pth_path=None) -> None:
        '''Initialize SimpleCNN
        
        Args:
            device (torch.device): Device to use
            version (str): Version of the model ('mobilenetv2-ssd-lite')
            input_size (tuple): Input size of the model (N, C, H, W)
            output_dir (str): Output directory
            pth_path (str): Path to the model checkpoint
        '''
        
        # --- Set parameters ---
        self.device = device
        self.input_size = input_size
        self.output_dir = output_dir
        
        # --- Load model ---
        if (version == 'mobilenetv2-ssd-lite'):
            self.net = create_mobilenetv2_ssd_lite(num_classes, width_mult=1.0)
        print(summary(self.net, input_size=input_size))
        
        # --- Transform ---
#        self.transform = transforms.Compose([
#            transforms.ToTensor(),
#        ])
        self.train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
        self.target_transform = MatchPrior(config.priors, config.center_variance,
                                    config.size_variance, 0.5)
        self.test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)        
        
    def predict(self, testloader, score_th=0.5):
        predictions = []
        targets = []
        processing_time = {
            'preprocessing': 0.0,
            'inference': 0.0,
        }
        self.net.to(self.device)
        with torch.no_grad():
            for (inputs, targets_, preprocessing_time_) in tqdm(testloader):
                inputs = [input.to(self.device) for input in inputs]
                self.net.eval()
                start = time.time()
                outputs = self.net(inputs)
                inference_time = time.time() - start
                for output in outputs:
                    valid_prediction = output['scores'] > score_th
                    predictions += [{key: output[key][valid_prediction].cpu().detach().numpy() for key in output.keys()}]
                    boxes = predictions[-1]['boxes']
                    predictions[-1]['boxes'] = np.array([boxes[:, 0], boxes[:, 1], boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]]).T
                targets += targets_
                processing_time['preprocessing'] += np.array(preprocessing_time_).sum()
                processing_time['inference'] += inference_time
            
            n_data = len(targets)
            processing_time['preprocessing'] /= n_data
            processing_time['inference'] /= n_data
        return predictions, targets, processing_time
    
    def evaluate(self, y_true, y_pred) -> dict:
        # --- T.B.D ---
        return
    
    def get_output_names(self) -> list:
        return ['boxes', 'scores', 'labels']

