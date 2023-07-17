import torch.nn as nn
import torch.nn.functional as F

import timm


class Densenet121(nn.Module):
  def __init__(self, num_features=5):
    super().__init__()
    self.model = timm.create_model("densenet121", pretrained=True)
    self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    self.model.classifier = nn.Linear(in_features=1024, out_features=num_features)
    
  def forward(self, x):
    x = self.model(x)
    return x