# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage10(torch.nn.Module):
    def __init__(self):
        super(Stage10, self).__init__()
        #self.layer1 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #self.layer2 = torch.nn.ReLU(inplace=True)
        #self.layer3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #self.layer4 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #self.layer5 = torch.nn.ReLU(inplace=True)
        self.layer26 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer27 = torch.nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        out26 = self.layer26(out0)
        out27 = self.layer27(out26)
        return out27
        #out1 = self.layer1(out0)
        #out2 = self.layer2(out1)
        #out3 = self.layer3(out2)
        #out4 = self.layer4(out3)
        #out5 = self.layer5(out4)
        #return out5

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
