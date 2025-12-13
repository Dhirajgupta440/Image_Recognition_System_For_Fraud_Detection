import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50(num_classes=2, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, num_classes)
    return model
