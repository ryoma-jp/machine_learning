from typing import Tuple
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchinfo import summary
from .yolo.yolo import YOLOX

from tqdm import tqdm
from pathlib import Path
from models.pytorch.pytorch_model_base import PyTorchModelBase

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
        
    def train(self, trainloader, epochs=10, optim_params=None, output_dir=None) -> None:
        """Train the model
        
        Args:
            trainloader (torch.utils.data.DataLoader): DataLoader for training
            epochs (int): Number of epochs
            optim_params (dict): Optimizer parameters
                - optim_params['optim'] (str): Optimizer name ['adamw', 'sgd', 'momentum', 'adagrad']
                    - adamw: AdamW optimizer
                        - lr: Learning rate
                        - wd: Weight decay
                    - sgd: SGD optimizer
                        - lr: Learning rate
                    - momentum: SGD with momentum optimizer
                        - lr: Learning rate
                        - momentum: Momentum
                    - adagrad: Adagrad optimizer
                        - lr: Learning rate
                        - lr_decay: Learning rate decay
                        - wd: Weight decay
            output_dir (str): Output directory to save the model
        """
        criterion = nn.CrossEntropyLoss()
        
        if (optim_params is None):
            optimizer = optim.AdamW(self.net.parameters())
        else:
            if (optim_params['optim'] == 'adamw'):
                lr = optim_params['lr'] if ('lr' in optim_params) else 0.001
                beta1 = optim_params['beta1'] if ('beta1' in optim_params) else 0.9
                beta2 = optim_params['beta2'] if ('beta2' in optim_params) else 0.999
                wd = optim_params['wd'] if ('wd' in optim_params) else 0.01
                optimizer = optim.AdamW(self.net.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)
            elif (optim_params['optim'] == 'sgd'):
                lr = optim_params['lr'] if ('lr' in optim_params) else 0.001
                optimizer = optim.SGD(self.net.parameters(), lr=lr)
            elif (optim_params['optim'] == 'momentum'):
                lr = optim_params['lr'] if ('lr' in optim_params) else 0.001
                momentum = optim_params['momentum'] if ('momentum' in optim_params) else 0.9
                optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
            elif (optim_params['optim'] == 'adagrad'):
                lr = optim_params['lr'] if ('lr' in optim_params) else 0.001
                lr_decay = optim_params['lr_decay'] if ('lr_decay' in optim_params) else 0.0
                wd = optim_params['wd'] if ('wd' in optim_params) else 0.0
                optimizer = optim.Adagrad(self.net.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=wd)
            elif (optim_params['optim'] == 'rmsprop'):
                lr = optim_params['lr'] if ('lr' in optim_params) else 0.01
                alpha = optim_params['alpha'] if ('alpha' in optim_params) else 0.99
                optimizer = optim.RMSprop(self.net.parameters(), lr=lr, alpha=alpha)
            elif (optim_params['optim'] == 'adadelta'):
                lr = optim_params['lr'] if ('lr' in optim_params) else 1.0
                rho = optim_params['rho'] if ('rho' in optim_params) else 0.9
                optimizer = optim.Adadelta(self.net.parameters(), lr=lr, rho=rho)
            elif (optim_params['optim'] == 'adam'):
                lr = optim_params['lr'] if ('lr' in optim_params) else 0.001
                beta1 = optim_params['beta1'] if ('beta1' in optim_params) else 0.9
                beta2 = optim_params['beta2'] if ('beta2' in optim_params) else 0.999
                wd = optim_params['wd'] if ('wd' in optim_params) else 0.01
                optimizer = optim.Adam(self.net.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)
            else:
                optimizer = optim.AdamW(self.net.parameters())
        
        # --- Caluculate first loss ---
        running_loss = 0.0
        for (data, preprocessing_time_) in tqdm(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        print(f'[EPOCH #0] loss: {running_loss/len(trainloader)}')
        
        # --- Training loop ---
        train_results = {
            'loss': [],
        }
        train_start = time.time()
        for epoch in range(epochs):
            running_loss = 0.0
            for (data, preprocessing_time_) in tqdm(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            epoch_loss = running_loss/len(trainloader)
            print(f'[EPOCH #{epoch+1}, elapsed time: {time.time()-train_start:.3f}[sec]] loss: {epoch_loss}')
            train_results['loss'].append(epoch_loss)
            
        # --- Save model ---
        if (output_dir is not None):
            if (not Path(output_dir).exists()):
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            model_path = Path(output_dir, 'model.pth')
            torch.save(self.net.state_dict(), model_path)
            self.convert_to_onnx(Path(output_dir, 'model.onnx'))
            self.convert_to_arm_compute_lib_via_tvm(Path(output_dir, 'arm_compute_library'))
        
        return train_results
    
    def predict(self, testloader, save_dir=None):
        self.net.to(self.device)
        input_tensor_names = []
        predictions = []
        labels = []
        processing_time = {
            'preprocessing': 0.0,
            'inference': 0.0,
        }
        
        if (save_dir is not None):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            input_tensor_dir = Path(save_dir, 'input_tensors')
            input_tensor_dir.mkdir(parents=True, exist_ok=True)
            sample_idx = 0
        with torch.no_grad():
            for (data, preprocessing_time_) in tqdm(testloader):
                inputs, labels_ = data
                inputs, labels_ = inputs.to(self.device), labels_.to(self.device)
                self.net.eval()
                start = time.time()
                outputs = self.net(inputs)
                inference_time = time.time() - start
                _, prediction = torch.max(outputs, 1)
                
                if (save_dir is not None):
                    for input_tensor in inputs:
                        sample_idx += 1
                        input_tensor_name = f'input_{sample_idx:08d}.npy'
                        input_tensor_names.append(input_tensor_name)
                        if (save_dir is not None):
                            np.save(Path(input_tensor_dir, input_tensor_name), input_tensor.to('cpu').detach().numpy())
                predictions.extend(prediction.to('cpu').detach().numpy().tolist().copy())
                labels.extend(labels_.to('cpu').detach().numpy().tolist().copy())
                processing_time['preprocessing'] += np.array(preprocessing_time_).sum()
                processing_time['inference'] += inference_time
            
        n_data = len(predictions)
        processing_time['preprocessing'] /= n_data
        processing_time['inference'] /= n_data
        
        if (save_dir is not None):
            inputs = pd.DataFrame({
                'input_tensor_names': input_tensor_names,
                'labels': labels,
            })
            inputs.to_csv(Path(input_tensor_dir, 'inputs.csv'), index=False)
            
            prediction_results = pd.DataFrame({
                'input_tensor_names': input_tensor_names,
                'predictions': predictions,
                'labels': labels,
            })
            prediction_results.to_csv(Path(save_dir, 'prediction_results.csv'), index=False)
        
        return predictions, labels, processing_time
    
    def evaluate(self, y_true, y_pred) -> dict:
        accuracy = accuracy_score(y_true, y_pred)
        cls_report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': cls_report,
        }
    