import torch
from torch import nn
from torchvision import models


class AgePredictor(nn.Module):
    def __init__(self, dropout_prob=0.):
        super(AgePredictor, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(1000, 3))

    def forward(self, x):
        x = self.resnet(x)
        return self.classifier(x)


def get_model(dropout_prob=0.):
    return AgePredictor(dropout_prob=dropout_prob)


def save_model(model, path):
    torch.save(model, path)


def load_model(path: str):
    return torch.load(path)
