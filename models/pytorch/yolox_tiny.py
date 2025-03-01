from typing import Tuple
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
#        self.net.to(self.device)
#        input_tensor_names = []
#        predictions = []
#        labels = []
#        processing_time = {
#            'preprocessing': 0.0,
#            'inference': 0.0,
#        }
#        
#        if (save_dir is not None):
#            Path(save_dir).mkdir(parents=True, exist_ok=True)
#            input_tensor_dir = Path(save_dir, 'input_tensors')
#            input_tensor_dir.mkdir(parents=True, exist_ok=True)
#            sample_idx = 0
#        with torch.no_grad():
#            for (data, preprocessing_time_) in tqdm(testloader):
#                inputs, labels_ = data
#                inputs, labels_ = inputs.to(self.device), labels_.to(self.device)
#                self.net.eval()
#                start = time.time()
#                outputs = self.net(inputs)
#                inference_time = time.time() - start
#                _, prediction = torch.max(outputs, 1)
#                
#                if (save_dir is not None):
#                    for input_tensor in inputs:
#                        sample_idx += 1
#                        input_tensor_name = f'input_{sample_idx:08d}.npy'
#                        input_tensor_names.append(input_tensor_name)
#                        if (save_dir is not None):
#                            np.save(Path(input_tensor_dir, input_tensor_name), input_tensor.to('cpu').detach().numpy())
#                predictions.extend(prediction.to('cpu').detach().numpy().tolist().copy())
#                labels.extend(labels_.to('cpu').detach().numpy().tolist().copy())
#                processing_time['preprocessing'] += np.array(preprocessing_time_).sum()
#                processing_time['inference'] += inference_time
#            
#        n_data = len(predictions)
#        processing_time['preprocessing'] /= n_data
#        processing_time['inference'] /= n_data
#        
#        if (save_dir is not None):
#            inputs = pd.DataFrame({
#                'input_tensor_names': input_tensor_names,
#                'labels': labels,
#            })
#            inputs.to_csv(Path(input_tensor_dir, 'inputs.csv'), index=False)
#            
#            prediction_results = pd.DataFrame({
#                'input_tensor_names': input_tensor_names,
#                'predictions': predictions,
#                'labels': labels,
#            })
#            prediction_results.to_csv(Path(save_dir, 'prediction_results.csv'), index=False)
        
        predictions = None
        labels = None
        processing_time = None
        return predictions, labels, processing_time
    
    def decode_predictions(self, predictions):
        boxes = []
        scores = []
        classes = []
        num_detections = 0
        threshold=0.5
        
        for i, detection in enumerate(predictions):
            if len(detection) == 0:
                continue
            for j in range(len(detection)):
                bbox = np.array(detection)[j][:4]
                score = np.array(detection)[j][4]
                class_id = np.argmax(np.array(detection)[j][5:])
                if score < threshold:
                    continue
                else:
                    boxes.append(bbox)
                    scores.append(score)
                    classes.append(class_id)
                    num_detections = num_detections + 1
        
        return {'detection_boxes': [boxes], 
                'detection_classes': [classes], 
                'detection_scores': [scores],
                'num_detections': [num_detections]}
    
    def evaluate(self, y_true, y_pred) -> dict:
        # --- T.B.D ---
        return None
