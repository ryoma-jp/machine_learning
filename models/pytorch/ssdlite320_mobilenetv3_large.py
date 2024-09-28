
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
from torchinfo import summary
from tqdm import tqdm
from models.pytorch.pytorch_model_base import PyTorchModelBase

class SSDLite320MobileNetv3Large(PyTorchModelBase):
    def __init__(self, device, input_size, output_dir='outputs', pth_path=None) -> None:
        '''Initialize SimpleCNN
        
        Args:
            device (torch.device): Device to use
            input_size (tuple): Input size of the model (N, C, H, W)
            output_dir (str): Output directory
            pth_path (str): Path to the model checkpoint
        '''
        
        # --- Set parameters ---
        self.device = device
        self.input_size = input_size
        self.output_dir = output_dir
        
        # --- Load model ---
        if (pth_path is None):
            self.net = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
            self.net.to(self.device)
            print(summary(self.net, input_size=input_size))
        else:
            # --- T.B.D ---
            self.net = None
        
        # --- Transform ---
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def predict(self, testloader, score_th=0.5, save_dir=None):
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

