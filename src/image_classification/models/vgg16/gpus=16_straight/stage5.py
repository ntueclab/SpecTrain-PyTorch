# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage5(torch.nn.Module):
    def __init__(self):
        super(Stage5, self).__init__()
        self.layer14 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer15 = torch.nn.ReLU(inplace=True)
        #self.layer1 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        out14 = self.layer14(out0)
        out15 = self.layer15(out14)
        return out15

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
