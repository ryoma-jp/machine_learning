from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv3_64 = nn.Conv2d(3, 64, 3, padding='same')
        self.bn2d64_1 = nn.BatchNorm2d(64)
        self.conv64_64 = nn.Conv2d(64, 64, 3, padding='same')
        self.bn2d64_2 = nn.BatchNorm2d(64)
        self.conv64_128 = nn.Conv2d(64, 128, 3, padding='same')
        self.bn2d128_1 = nn.BatchNorm2d(128)
        self.conv128_128 = nn.Conv2d(128, 128, 3, padding='same')
        self.bn2d128_2 = nn.BatchNorm2d(128)
        self.conv128_256 = nn.Conv2d(128, 256, 3, padding='same')
        self.bn2d256_1 = nn.BatchNorm2d(256)
        self.conv256_256 = nn.Conv2d(256, 256, 3, padding='same')
        self.bn2d256_2 = nn.BatchNorm2d(256)

        self.fc256_128 = nn.Linear(256, 128)
        self.bn1d128_1 = nn.BatchNorm1d(128)
        self.fc128_64 = nn.Linear(128, 64)
        self.bn1d64_1 = nn.BatchNorm1d(64)
        self.fc64_10 = nn.Linear(64, 10)

        self.relu = nn.ReLU()
        self.dropout2d = nn.Dropout(0.5)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.MaxPool2d((2, 2), padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
        x = x.view(-1, 256)
        x = self.dropout(self.bn1d128_1(self.relu(self.fc256_128(x))))
        x = self.dropout(self.bn1d64_1(self.relu(self.fc128_64(x))))
        
        x = self.softmax(self.fc64_10(x))
        return x
    

class SimpleCNN():
    def __init__(self, device) -> None:
        self.device = device
        self.net = Net()
        self.net.to(self.device)
        
    def train(self, trainloader, epochs=10, lr=0.0001, wd=0.01, output_dir=None) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        
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
            print(f'[EPOCH #{epoch+1}] loss: {running_loss/len(trainloader)}')
            
        # --- Save model ---
        if (output_dir is not None):
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
    