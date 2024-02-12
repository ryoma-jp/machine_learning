import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class SimpleCNN():
    def __init__(self) -> None:
        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        
    def train(self, trainloader, epochs=2, output_dir=None) -> None:
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                inputs, labels = data
                
                self.optimizer.zero_grad()
                
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if ((i+1) % 1000 == 0):
                    print(f'[EPOCH #{epoch+1}, Iter #{i+1}] loss: {running_loss/1000}')
                    running_loss = 0.0
                    
        if (output_dir is not None):
            model_path = Path(output_dir, 'model.pth')
            torch.save(self.net.state_dict(), model_path)
            
    def predict(self, testloader) -> torch.Tensor:
        return self.net(testloader)
    
    def evaluate(self, testloader) -> float:
        pass
    