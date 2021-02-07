# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage12(torch.nn.Module):
    def __init__(self):
        super(Stage12, self).__init__()
        #self.layer1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #self.layer2 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer30 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer31 = torch.nn.ReLU(inplace=True)
        self.layer32 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer33 = torch.nn.AdaptiveAvgPool2d((7, 7))
        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        #out1 = self.layer1(out0)
        #out2 = self.layer2(out1)
        out30 = self.layer30(out0)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)
        out33 = self.layer33(out32)
        out34 = out33.size(0)
        out35 = out33.view(out34, -1)
        return out35
        #return out32

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
