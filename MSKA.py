import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict


class MSKAattention(nn.Module):

    def __init__(self, channel, kernels=[1, 3, 5]):
        super(MSKAattention, self).__init__()

        # Adaptive
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2)),  # H_out=H_in
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )

        # SA
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // 2, 1, 1))  # (1, 4, 1, 1)
        self.cbias = Parameter(torch.ones(1, channel // 2, 1, 1))  # (1, 4, 1, 1)
        self.sweight = Parameter(torch.zeros(1, channel // 2, 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // 2, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // 2, channel // 2)

        # utils
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        fuss = []
        # split
        for conv in self.convs:
            conv_outs.append(conv(x))
        # fuse
        U = sum(conv_outs)  # bs,c,h,w

        b, c, h, w = U.shape

        x = x.reshape(b, -1, h, w)
        assert (x.shape[1] > 1) and (x.shape[1] % 2 == 0),
        x_0, x_1 = x.chunk(2, dim=1)

        xn = self.avg_pool(x_0)  # (64, 4, 1, 1)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)

        return out
