import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from typing import Tuple
from torchvision import transforms
from torchinfo import summary
from .yolo.yolo import YOLOX

from tqdm import tqdm
from pathlib import Path
from models.pytorch.pytorch_model_base import PyTorchModelBase
from .utils.transforms import PaddingImageTransform

class YOLOX_Tiny(PyTorchModelBase):
    def __init__(self, device, input_size, num_classes, output_dir='outputs', pth_path=None) -> None:
        '''Initialize SimpleCNN
        
        Args:
            device (torch.device): Device to use
            input_size (tuple): Input size of the model (N, C, H, W)
            num_classes (int): Number of classes
        '''
        self.device = device
        net_input_size = input_size[1:]
        self.input_size = [1] + list(input_size[1:])
        self.output_dir = output_dir

        # --- Load model ---
        name = 'yolox_tiny'
        backbone = None
        head = None
        num_classes = num_classes
        self.net = YOLOX(name, backbone, head, num_classes)
        if (pth_path is not None):
            self.net.load_state_dict(torch.load(pth_path))
        
        self.net.to(self.device)
        print(summary(self.net, input_size=input_size))

        # --- Transform ---
        height, width = net_input_size[1:]
        self.transform = transforms.Compose([
            PaddingImageTransform((height, width)),
            transforms.ToTensor()
        ])

    def train(self, trainloader, epochs=10, optim_params=None, output_dir=None) -> None:
        # --- T.B.D ---
        train_results = None
        return train_results
    
    def predict(self, testloader, save_dir=None):
        self.net.to(self.device)
        input_tensor_names = []
        predictions = []
        targets = []
        processing_time = {
            'preprocessing': 0.0,
            'inference': 0.0,
            'postprocessing': 0.0,
        }
        
        if (save_dir is not None):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            input_tensor_dir = Path(save_dir, 'input_tensors')
            input_tensor_dir.mkdir(parents=True, exist_ok=True)
            sample_idx = 0

        with torch.no_grad():
            self.net.eval()
            for (inputs, target, preprocessing_time_) in tqdm(testloader):
                start = time.time()
                # convert tuple to torch.Tensor
                inputs = torch.stack(inputs).to(self.device)

                # inference
                inputs = inputs * 255.0
                outputs = self.net(inputs)
                print(outputs)
                inference_time = time.time() - start
                
                start = time.time()
                for output in outputs.cpu().numpy().tolist():
                    predictions.append(self.decode_predictions(output))
                postprocessing_time = time.time() - start

                if (save_dir is not None):
                    for input_tensor in inputs:
                        sample_idx += 1
                        input_tensor_name = f'input_{sample_idx:08d}.npy'
                        input_tensor_names.append(input_tensor_name)
                        if (save_dir is not None):
                            np.save(Path(input_tensor_dir, input_tensor_name), input_tensor.to('cpu').detach().numpy())
                targets += target
                processing_time['preprocessing'] += np.array(preprocessing_time_).sum()
                processing_time['inference'] += inference_time
                processing_time['postprocessing'] += postprocessing_time

                # inference
                #outputs = self.net(inputs * 255.0)
                inference_time = time.time() - start
                
                start = time.time()
                for output in outputs.cpu().numpy().tolist():
                    predictions.append(self.decode_predictions(output))
                postprocessing_time = time.time() - start

                if (save_dir is not None):
                    for input_tensor in inputs:
                        sample_idx += 1
                        input_tensor_name = f'input_{sample_idx:08d}.npy'
                        input_tensor_names.append(input_tensor_name)
                        if (save_dir is not None):
                            np.save(Path(input_tensor_dir, input_tensor_name), input_tensor.to('cpu').detach().numpy())
                targets += target
                processing_time['preprocessing'] += np.array(preprocessing_time_).sum()
                processing_time['inference'] += inference_time
                processing_time['postprocessing'] += postprocessing_time
            
        n_data = len(predictions)
        processing_time['preprocessing'] /= n_data
        processing_time['inference'] /= n_data
        processing_time['postprocessing'] /= n_data
        
        return predictions, targets, processing_time
    
    def decode_predictions(self, predictions):
        num_detections = 0
        threshold=0.5
        
        detections = np.array(predictions)
        bboxes = detections[:, :4]
        scores = detections[:, 4]
        class_ids = np.argmax(detections[:, 5:], axis=1)
        mask = scores >= threshold
        bboxes = bboxes[mask]
        # --- convert [center_x, center_y, width, height] to [x, y, w, h]
        bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x = center_x - width / 2
        bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y = center_y - height / 2

        scores = scores[mask]
        classes = class_ids[mask]
        num_detections += mask.sum()

        return {
            'boxes': bboxes, 
            'labels': classes, 
            'scores': scores,
            'num_detections': num_detections}
    
    def evaluate(self, y_true, y_pred) -> dict:
        # --- T.B.D ---
        return None

    def get_output_names(self) -> list:
        return ['boxes', 'scores', 'labels']

