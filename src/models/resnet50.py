"""
===============================================================
ResNet50 Model
Transfer Learning for Mammography Classification
===============================================================
"""

import torch.nn as nn
from torchvision import models

from config.config import PRETRAINED


class ResNet50Model(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        # Load pretrained ResNet50
        if PRETRAINED:
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT
            )
        else:
            self.model = models.resnet50(weights=None)

        # Freeze backbone initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last residual block
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace classifier
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(

            nn.Linear(in_features, 512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(512, num_classes)

        )

    def forward(self, x):

        return self.model(x)