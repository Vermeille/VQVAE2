import torch
import torch.nn as nn
import torch.nn.functional as F

from vq import VQ


def Conv(in_ch, out_ch, ks):
    return nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2)


def ConvBNRelu(in_ch, out_ch, ks):
    return nn.Sequential(
            Conv(in_ch, out_ch, ks),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class ResBlk(nn.Module):
    def __init__(self, ch):
        super(ResBlk, self).__init__()
        self.go = nn.Sequential(
                ConvBNRelu(ch, ch * 2, 3),
                ConvBNRelu(ch * 2, ch, 3),
            )

    def forward(self, x):
        return x + self.go(x)


class Encoder(nn.Module):
    def __init__(self, arch, hidden=64):
        super(Encoder, self).__init__()
        layers = [
            Conv(3, hidden, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden),
        ]

        b_vq = 128
        for l in arch:
            if l == 'r':
                layers.append(ResBlk(hidden))
            elif l == 'q':
                layers.append(VQ(hidden, b_vq, dim=1))
            elif l == 'p':
                layers.append(nn.AvgPool2d(3, 2, 1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, ret_idx=False):
        qs = []
        idxs = []
        for m in self.layers:
            if isinstance(m, VQ):
                x, idx = m(x)
                qs.append(x.detach())
                idxs.append(idx)
            else:
                x = m(x)

        if ret_idx:
            return qs, idxs
        else:
            return qs


class Decoder(nn.Module):
    def __init__(self, arch, hidden=64):
        super(Decoder, self).__init__()
        layers = []

        for l in arch:
            if l == 'r':
                layers.append(ResBlk(hidden))
            elif l == 'u':
                layers.append(nn.UpsamplingNearest2d(scale_factor=2))
            elif l == 'c':
                layers.append(nn.Conv2d(hidden*2, hidden, 1))

        layers += [
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Conv(hidden, 3, 3),
            nn.Sigmoid()
        ]

        self.layers = nn.ModuleList(layers)

    def forward(self, qs):
        x = qs.pop()
        for m in self.layers:
            if isinstance(m, nn.Conv2d) and m.weight.shape[3] == 1:
                q = qs.pop()
                x = torch.cat([x, q], dim=1)
            x = m(x)

        return x


def AE(enc, dec):
    return AE_initialize(nn.Sequential(Encoder(enc), Decoder(dec)))


def AE_initialize(ae):
    for m in ae.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return ae

def baseline():
    return AE('rrprrpqrrq', 'rrcurur')

def baseline_64():
    return AE('rrpqrrpq', 'rrucrru')
