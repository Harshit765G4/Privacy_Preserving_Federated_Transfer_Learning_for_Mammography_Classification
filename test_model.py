import torch

from src.models.resnet50 import ResNet50Model

model = ResNet50Model()

print("=" * 60)

print(model)

dummy = torch.randn(2, 3, 512, 512)

output = model(dummy)

print()

print("Output Shape :", output.shape)

print("=" * 60)