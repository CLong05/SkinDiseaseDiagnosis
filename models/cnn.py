import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x = self.bn1(self.dropout(x))
        x = torch.relu(self.fc1(x))
        x = self.bn2(self.dropout(x))
        return self.fc2(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.drop_channel = nn.Dropout2d(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.mlp = Mlp(4 * 4 * 256, 100, 40)

    def forward(self, x):
        x = torch.relu(self.pool(self.conv1(x)))  # 128 -> 64
        x = self.bn1(x)
        x = torch.relu(self.pool(self.conv2(x)))  # 64 -> 32
        x = self.drop_channel(x)
        x = self.bn2(x)
        x = torch.relu(self.pool(self.conv3(x)))  # 32 -> 16
        x = self.bn3(x)
        x = torch.relu(self.pool(self.conv4(x)))  # 16 -> 8
        x = self.drop_channel(x)
        x = self.bn4(x)
        x = torch.relu(self.pool(self.conv5(x)))  # 8 -> 4
        x = x.view(x.shape[0], -1)
        return self.mlp(x)

if __name__ == '__main__':
    f = CNN()
    x = torch.rand(16, 3, 128, 128)
    f(x)
