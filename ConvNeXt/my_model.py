import torch
import torch.nn as nn
from torchvision import models


def my_convnext(num_class):
    model = models.convnext_tiny(False)
    in_channel = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_channel, num_class)

    return model

def my_resnet(num_class):
    model = models.resnet101(True)
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, num_class)

    return model


if __name__ == '__main__':
    model = models.resnet101(False)
    print(model)