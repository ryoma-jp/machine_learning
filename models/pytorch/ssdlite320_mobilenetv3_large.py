
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from tqdm import tqdm

class SSDLite320MobileNetv3Large():
    def __init__(self, device, input_size, pth_path=None) -> None:
        '''Initialize SimpleCNN
        
        Args:
            device (torch.device): Device to use
            input_size (tuple): Input size of the model (N, C, H, W)
            pth_path (str): Path to the model checkpoint
        '''
        
        # --- Set device ---
        self.device = device
        
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
        
    def predict(self, testloader, score_th=0.5):
        predictions = []
        targets = []
        with torch.no_grad():
            for (inputs, targets_) in tqdm(testloader):
                inputs = [input.to(self.device) for input in inputs]
                self.net.eval()
                outputs = self.net(inputs)
                for output in outputs:
                    valid_prediction = output['scores'] > score_th
                    predictions += [{key: output[key][valid_prediction].cpu().detach().numpy() for key in output.keys()}]
                    boxes = predictions[-1]['boxes']
                    predictions[-1]['boxes'] = np.array([boxes[:, 0], boxes[:, 1], boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]]).T
                targets += targets_
        return predictions, targets
    
    def evaluate(self, y_true, y_pred) -> dict:
        # --- T.B.D ---
        return
    