from typing import Tuple
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

class Net(nn.Module):
    def __init__(self, input_size, classes_num) -> None:
        '''Initialize Net
        
        Args:
            input_size (tuple): Input size of the model (C, H, W)
            classes_num (int): Number of classes
        '''
        super().__init__()
        channels, height, width = input_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        self.fc2= nn.Sequential(
            nn.Linear(4096, classes_num))

        for m in self.modules():
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


    def forward(self, x) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    

class VGG16():
    def __init__(self, device, input_size, num_classes) -> None:
        '''Initialize VGG16
        
        Args:
            device (torch.device): Device to use
            input_size (tuple): Input size of the model (N, C, H, W)
            num_classes (int): Number of classes
        '''
        self.device = device
        net_input_size = input_size[1:]
        self.net = Net(net_input_size, num_classes)
        self.net.to(self.device)
        print(summary(self.net, input_size=input_size))
        
    def train(self, trainloader, epochs=10, lr=0.0001, wd=0.01, output_dir=None) -> None:
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay=wd)
        
        # --- Caluculate first loss ---
        period = len(trainloader)//1000
        running_loss = 0.0
        for step, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            if (step % period == 0):
                print(f'[EPOCH #0, step #{step}] loss: {running_loss/(step+1)}')
        print(f'[EPOCH #0] loss: {running_loss/len(trainloader)}')
        
        # --- Training loop ---
        train_start = time.time()
        for epoch in range(epochs):
            running_loss = 0.0
            for step, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if (step % period == 0):
                    print(f'[EPOCH #{epoch+1}, step #{step}] loss: {running_loss/(step+1)}')
            print(f'[EPOCH #{epoch+1}, elapsed time: {time.time()-train_start:.3f}[sec]] loss: {running_loss/len(trainloader)}')
            
        # --- Save model ---
        if (output_dir is not None):
            if (not Path(output_dir).exists()):
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            model_path = Path(output_dir, 'model.pth')
            torch.save(self.net.state_dict(), model_path)
            
    def predict(self, testloader) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        labels = []
        with torch.no_grad():
            for data in testloader:
                inputs, labels_ = data
                inputs, labels_ = inputs.to(self.device), labels_.to(self.device)
                outputs = self.net(inputs)
                _, prediction = torch.max(outputs, 1)
                predictions.extend(prediction.to('cpu').detach().numpy().tolist().copy())
                labels.extend(labels_.to('cpu').detach().numpy().tolist().copy())
        return predictions, labels
    
    def evaluate(self, y_true, y_pred) -> dict:
        accuracy = accuracy_score(y_true, y_pred)
        cls_report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': cls_report,
        }
    