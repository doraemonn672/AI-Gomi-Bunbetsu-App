import torch
from torch import nn

class WasteCNN(nn.Module):
    def __init__(self, channels=16, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels, channels*2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels*2, channels*4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(channels*4, num_classes))
    def forward(self, x):
        return self.classifier(self.features(x))
