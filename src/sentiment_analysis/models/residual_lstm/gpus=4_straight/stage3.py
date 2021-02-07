# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .model import Transition, Bottleneck



class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.emb_size = 128
        self.hid_size = 128

        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, batch_first=True)  
        self.projection_matrix = nn.Linear(self.hid_size, self.emb_size)
        self.output = nn.Linear(self.emb_size, 2)
        
        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        shortcut = out0
        output, (h_n, c_n) = self.LSTM(out0)
        output = self.projection_matrix(output)
        output += shortcut
        output = self.output(output.mean(dim=1))

        return output
    

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
