# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        
        # MAX_WORDS = 10000  # imdb’s vocab_size 即词汇表大小
        # MAX_LEN = 200      # max length
        # BATCH_SIZE = 256
        # EMB_SIZE = 128   # embedding size
        # HID_SIZE = 128   # lstm hidden size
        self.max_words = 10000
        self.emb_size = 128
        self.hid_size = 128

        self.embeddings = nn.Embedding(self.max_words, self.emb_size)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, batch_first=True)  
        self.projection_matrix = nn.Linear(self.hid_size, self.emb_size)
        
        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        # print(out0.size())
        output = self.embeddings(out0)
        
        # for i in range(self.num_blocks):
        shortcut = output
        output, (h_n, c_n) = self.LSTM(output)
        output = self.projection_matrix(output)
        output += shortcut
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
