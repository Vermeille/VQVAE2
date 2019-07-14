import torch
import torch.nn as nn
import torch.nn.functional as F

from torchelie.nn import VQ

GN32 = lambda x, affine: nn.GroupNorm2d


def Conv(in_ch, out_ch, ks):
    return nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2)


def ConvBNRelu(in_ch, out_ch, ks):
    return nn.Sequential(
            Conv(in_ch, out_ch, ks),
            nn.BatchNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=False)
        )


class ResBlk(nn.Module):
    def __init__(self, ch):
        super(ResBlk, self).__init__()
        self.go = nn.Sequential(
                ConvBNRelu(ch, ch, 3),
                ConvBNRelu(ch, ch, 3),
            )

    def forward(self, x):
        return self.go(x) + x


class Encoder(nn.Module):
    def __init__(self, arch, hidden=128, codebook_size=256):
        super(Encoder, self).__init__()
        layers = [
            Conv(3, hidden, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden, affine=True),
        ]

        b_vq = codebook_size
        for l in arch:
            if l == 'r':
                layers.append(ResBlk(hidden))
            elif l == 'q':
                layers.append(VQ(hidden, b_vq, dim=1))
            elif l == 'p':
                layers.append(nn.AvgPool2d(3, 2, 1))

        self.layers = nn.ModuleList(layers)

    def forward(self, dat, ret_idx=False):
        x, y = dat
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
    def __init__(self, arch, hidden=128):
        super(Decoder, self).__init__()
        layers = []

        for l in arch:
            if l == 'r':
                layers.append(ResBlk(hidden))
            elif l == 'u':
                layers.append(nn.UpsamplingNearest2d(scale_factor=2))
            elif l == 'c':
                layers.append(nn.Conv2d(hidden*2, hidden, 1))
            elif l == 'n':
                layers.append(Noise(hidden))

        layers += [
            nn.BatchNorm2d(hidden, affine=True),
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


def AE(enc, dec, hidden=64):
    return AE_initialize(nn.Sequential(Encoder(enc, hidden), Decoder(dec,
        hidden)))


def AE_initialize(ae):
    for m in ae.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', a=0.2)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return ae

def baseline(sz, noise=False):
    if sz == 128:
        if noise:
            return baseline_128_n()
        else:
            return baseline_128()
    if sz == 64:
        return baseline_64()

def baseline_64():
    return AE('rrpqrrpq', 'rrucrrur')

def baseline_128_n():
    return AE('rrprrpqrrpq', 'rnrucnrrurnrunr')

def baseline_128():
    return AE('rprpqrpq', 'rrucrrurrur')

def baseline_256():
    return AE('rprprprpq', 'rurururur')

def baseline_256_2l():
    return AE('rprprpqrpq', 'rrucrrurrurur')
