import torch.nn as nn
import torch.nn.functional as F

class ThumbnailCNN(nn.Module):
    def __init__(self):
        super(ThumbnailCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(32 * 174 * 310, 128)  # adjust based on output dims
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
