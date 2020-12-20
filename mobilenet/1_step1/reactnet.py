import torch
import torch.nn as nn

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

class reactnet(nn.Module):
    def __init__(self, NUM_CLASSES=200):
        super(reactnet, self).__init__()
        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i==0:
                self.feature.append(firstconv3x3(3, stage_out_channel[0], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i]:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1))
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, NUM_CLASSES)

    def forward(self, x):
        # TO-DO
        for i, block in enumerate(self.feature):
            print("x.shape(): {}".format(x.shape()))
            x = block(x)
        x = self.pool1(x)
        print("x.shape(): {}".format(x.shape()))
        x = x.view(x.size(0), 1)
        print("x.shape(): {}".format(x.shape()))
        x = self.fc(x)
        print("x.shape(): {}".format(x.shape()))

class firstconv3x3(nn.Module):
    def __init__(self, inp, outp, stride):
        super(firstconv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inp, out_channels=outp, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outp)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inp, outp, stride):
        super(BasicBlock, self).__init__()
        # TO-DO
        return
    def forward(self, x):
        # TO-DO
        return