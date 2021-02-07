# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage4(torch.nn.Module):
    def __init__(self):
        super(Stage4, self).__init__()
        self.layer12 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer13 = torch.nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        out12 = self.layer12(out0)
        out13 = self.layer13(out12)
        return out13

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
