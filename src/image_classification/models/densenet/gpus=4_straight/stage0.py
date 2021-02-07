# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Transition, Bottleneck

class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.growth_rate = growth_rate = 32
        num_planes = 2*growth_rate
        block = Bottleneck
        nblocks = [6,12,24,16]
        reduction=0.5
        num_classes=10

        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)

        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        out2 = self.conv1(out0)
        out3 = self.dense1(out2)
        out4 = self.trans1(out3)
        return out4
    
    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)
    
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
