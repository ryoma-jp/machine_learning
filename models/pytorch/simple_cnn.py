from typing import Tuple
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchinfo import summary

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from models.pytorch.pytorch_model_base import PyTorchModelBase

class Net(nn.Module):
    def __init__(self, input_size, classes_num) -> None:
        '''Initialize Net
        
        Args:
            input_size (tuple): Input size of the model (C, H, W)
            classes_num (int): Number of classes
        '''
        super().__init__()
        channels, height, width = input_size
        kernel_size = (3, 3)
        padding = (1, 1)
        
        self.conv3_64 = nn.Conv2d(channels, 64, kernel_size=kernel_size, padding=padding)
        self.bn2d64_1 = nn.BatchNorm2d(64)
        self.conv64_64 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.bn2d64_2 = nn.BatchNorm2d(64)
        self.conv64_128 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        self.bn2d128_1 = nn.BatchNorm2d(128)
        self.conv128_128 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.bn2d128_2 = nn.BatchNorm2d(128)
        self.conv128_256 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding)
        self.bn2d256_1 = nn.BatchNorm2d(256)
        self.conv256_256 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding)
        self.bn2d256_2 = nn.BatchNorm2d(256)

        self.flatten_nodes = 256 * (height//16) * (width//16)
        self.fcn_512 = nn.Linear(self.flatten_nodes, 512)
        self.bn1d512_1 = nn.BatchNorm1d(512)
        self.fc512_128 = nn.Linear(512, 128)
        self.bn1d128_1 = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, classes_num)

        self.relu = nn.ReLU()
        self.dropout2d = nn.Dropout(0.5)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.MaxPool2d((2, 2), padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((height//16, width//16))

    def forward(self, x) -> torch.Tensor:
        x = self.bn2d64_1(self.relu(self.conv3_64(x)))
        x = self.bn2d64_2(self.relu(self.conv64_64(x)))
        x = self.dropout2d(self.pool(x))
        
        x = self.bn2d128_1(self.relu(self.conv64_128(x)))
        x = self.bn2d128_2(self.relu(self.conv128_128(x)))
        x = self.dropout2d(self.pool(x))
        
        x = self.bn2d256_1(self.relu(self.conv128_256(x)))
        x = self.bn2d256_2(self.relu(self.conv256_256(x)))
        x = self.pool(x)
        
        x = self.avgpool(x)
        x = x.view(-1, self.flatten_nodes)
        x = self.dropout(self.bn1d512_1(self.relu(self.fcn_512(x))))
        x = self.dropout(self.bn1d128_1(self.relu(self.fc512_128(x))))
        
        x = self.softmax(self.fc_out(x))
        return x
    

class SimpleCNN(PyTorchModelBase):
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
        self.net = Net(net_input_size, num_classes)
        
        if (pth_path is None) or (not Path(pth_path).exists()):
            if (not Path(pth_path).exists()):
                print(f'[WARNING] {pth_path} is not found. Initialize the model with random values.')
                
            for m in self.net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        else:
            self.net.load_state_dict(torch.load(pth_path))
            
        self.net.to(self.device)
        print(summary(self.net, input_size=input_size))
        
        # --- Transform ---
        resize = input_size[2:]
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

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
        for data in trainloader:
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
            for data in trainloader:
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
    
    def predict(self, testloader) -> Tuple[np.ndarray, np.ndarray]:
        self.net.to(self.device)
        predictions = []
        labels = []
        processing_time = {
            'preprocessing': 0.0,
            'inference': 0.0,
        }
        with torch.no_grad():
            for (data, preprocessing_time_) in tqdm(testloader):
                inputs, labels_ = data
                inputs, labels_ = inputs.to(self.device), labels_.to(self.device)
                self.net.eval()
                start = time.time()
                outputs = self.net(inputs)
                inference_time = time.time() - start
                _, prediction = torch.max(outputs, 1)
                predictions.extend(prediction.to('cpu').detach().numpy().tolist().copy())
                labels.extend(labels_.to('cpu').detach().numpy().tolist().copy())
                processing_time['preprocessing'] += np.array(preprocessing_time_).sum()
                processing_time['inference'] += inference_time
            
        n_data = len(predictions)
        processing_time['preprocessing'] /= n_data
        processing_time['inference'] /= n_data
        
        return predictions, labels, processing_time
    
    def evaluate(self, y_true, y_pred) -> dict:
        accuracy = accuracy_score(y_true, y_pred)
        cls_report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': cls_report,
        }
    