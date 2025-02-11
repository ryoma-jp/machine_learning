
import onnxruntime
import torchvision.transforms as transforms
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from models.pytorch.ssdlite320_mobilenetv3_large import SSDLite320MobileNetv3Large as PyTorchSSDLite320MobileNetv3Large

class SSDLite320MobileNetv3Large():
    def __init__(self, device, input_size, output_dir='outputs', onnx_path=None) -> None:
        '''Initialize
        
        Args:
            device (torch.device): Device to use
            input_size (tuple): Input size of the model (N, C, H, W)
            output_dir (str): Output directory
            onnx_path (str): Path to the ONNX model
        '''
        
        # --- Set parameters ---
        self.device = device
        self.input_size = input_size
        self.output_dir = output_dir
        
        # --- Load model ---
        if (onnx_path is None):
            input_size = [1, 3, 320, 320]
            model = PyTorchSSDLite320MobileNetv3Large(self.device, input_size)
            onnx_path = Path(self.output_dir, 'model.onnx')
            model.convert_to_onnx(output_file=onnx_path, output_names=model.get_output_names())
            
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.session.get_modelmeta()
        input_names = [input.name for input in self.session.get_inputs()]
        self.boxes_key, self.scores_key, self.classes_key = [output.name for output in self.session.get_outputs()]
        print(f'Input names: {input_names}')
        print(f'Output names: {self.boxes_key}, {self.scores_key}, {self.classes_key}')
        
        # --- Transform ---
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size[2:], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        ])
        
    def predict(self, testloader, score_th=0.5, save_dir=None):
        predictions = []
        targets = []
        processing_time = {
            'preprocessing': 0.0,
            'inference': 0.0,
        }
        for (inputs, targets_, preprocessing_time_) in tqdm(testloader):
            start = time.time()
            inputs = np.array([input.numpy() for input in inputs])
            session_input = {
                self.session.get_inputs()[0].name: inputs,
            }
            results = self.session.run([], session_input)
            inference_time = time.time() - start
            session_output = {
                self.session.get_outputs()[i].name: results[i] for i in range(len(results))
            }
            
            valid_prediction = session_output[self.scores_key] > score_th
            predictions += [{key: session_output[key][valid_prediction] for key in [self.boxes_key, self.scores_key, self.classes_key]}]
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
    

