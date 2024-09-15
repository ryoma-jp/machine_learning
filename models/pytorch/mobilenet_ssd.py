
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
from torchinfo import summary
from tqdm import tqdm
from models.pytorch.pytorch_model_base import PyTorchModelBase
from .mobilenet_ssd_models.mobilenetv2_ssd_lite import create_mobilenetv2_ssd_lite

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
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
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

