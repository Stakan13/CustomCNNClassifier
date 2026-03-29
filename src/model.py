import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CNN_custom(nn.Module):
    
    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.5):
        super(CNN_custom, self).__init__()
        
        # Сверточные блоки
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.30)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.30)
        )
        
        # Полностью связанные слои
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 9 * 9, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.fc3 = nn.Linear(128, num_classes)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(-1, 256 * 9 * 9)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        conv_layers = {
            'conv1': self.conv1,
            'conv2': self.conv2,
            'conv3': self.conv3,
            'conv4': self.conv4
        }
        
        if layer_name not in conv_layers:
            raise ValueError(f"Layer {layer_name} not found")
        
        return conv_layers[layer_name](x)


def create_model(num_classes: int = 6, pretrained: bool = False) -> CNN_custom:
    model = CNN_custom(num_classes=num_classes)
    
    if pretrained:
        pass
    
    return model
