import torch
import torch.nn as nn

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

class reactnet(nn.Module):
    def __init__(self):
        super(reactnet, self).__init__()
        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            # first layer
            if i==0:
                # TO-DO
                return
            # reduction block
            elif stage_out_channel[i-1] != stage_out_channel[i]:
                # TO-DO
                return
            # normal block
            else:
                # TO-DO
                return
    def forward(self, x):
        # TO-DO
        return

class firstconv3x3(nn.Module):
    def __init__(self):
        # TO-DO
        return
    def forward(self, x):
        # TO-DO
        return