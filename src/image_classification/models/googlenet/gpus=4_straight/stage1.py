# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Inception


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        
        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        out = self.b3(out0)
        out = self.maxpool(out)
        # out = self.a4(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
