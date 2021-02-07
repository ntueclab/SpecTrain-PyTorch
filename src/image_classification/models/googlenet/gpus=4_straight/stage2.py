# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Inception


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        # self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        # self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)

        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        out = self.a4(out0)
        out = self.b4(out)
        # out = self.c4(out)
        # out = self.d4(out)
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
