import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_chan, out_chan, ks, center):
        super(MaskedConv2d, self).__init__(in_chan, out_chan, (ks // 2 + 1, ks), padding=0)
        self.register_buffer('mask', torch.ones(ks // 2 + 1, ks))
        self.mask[-1, ks // 2 + (1 if center else 0):] = 0

    def forward(self, x):
        self.weight_orig = self.weight
        del self.weight
        self.weight = self.weight_orig * self.mask
        ks = self.weight.shape[-1]

        x = F.pad(x, (ks // 2, ks // 2, ks // 2, 0))
        res = super(MaskedConv2d, self).forward(x)

        self.weight = self.weight_orig
        return res


def MConvBNrelu(in_ch, out_ch, ks, center=True):
    return nn.Sequential(
            MaskedConv2d(in_ch, out_ch, ks, center=center),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class ResBlk(nn.Module):
    def __init__(self, ch, ks):
        super(ResBlk, self).__init__()
        self.go = nn.Sequential(
                MConvBNrelu(ch, ch * 2, ks),
                MConvBNrelu(ch * 2, ch, ks),
        )

    def forward(self, x):
        return x + self.go(x)


class PixCNNBase(nn.Module):
    def __init__(self, in_ch, hid, out_ch, sz):
        super(PixCNNBase, self).__init__()
        self.sz = sz
        self.lin = MConvBNrelu(in_ch, hid, 5, center=False)
        self.bias = nn.Parameter(torch.zeros(hid, *sz))

        self.l1 = ResBlk(hid, 3)
        self.l2 = ResBlk(hid, 3)
        self.l3 = ResBlk(hid, 3)
        self.l4 = ResBlk(hid, 3)
        self.l5 = ResBlk(hid, 3)
        self.l6 = ResBlk(hid, 3)

        self.lout = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hid),
            MaskedConv2d(hid, out_ch, 3, center=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                #m.weight.data *= 2
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.lin(x) + self.bias

        x1 = self.l1(x) #1
        x2 = self.l2(F.interpolate(x1, scale_factor=0.5)) #2
        x3 = self.l3(F.interpolate(x2, scale_factor=0.5)) #3

        x4 = self.l4(x3) #3
        x5 = self.l5(F.interpolate(x4, scale_factor=2) + x2)
        x6 = self.l6(F.interpolate(x5, scale_factor=2) + x1)
        return self.lout(x6)


class CodebookCNN(PixCNNBase):
    def __init__(self, in_ch, hid, out_ch, sz):
        super(CodebookCNN, self).__init__(in_ch, hid, out_ch, sz)

    def sample(self, codebook, temp, N):
        img = torch.zeros(N, self.bias.shape[0], *self.sz, device=self.bias.device)
        return self.sample_(img, codebook, temp)


    def sample_cond(self, cond, codebook, temp):
        img = torch.zeros(cond.shape[0], self.bias.shape[0], *self.sz, device=self.bias.device)
        print(img.shape, cond.shape)
        cond_rsz = F.interpolate(cond, size=img.shape[2:], mode='nearest')
        img = torch.cat([img, cond_rsz], dim=1)
        return self.sample_(img, codebook, temp)[:, :cond.shape[1]]


    def sample_(self, img, codebook, temp=0):
        self.eval()
        with torch.no_grad():
            for l in range(self.sz[0]):
                for c in range(self.sz[0]):
                    log_prob = self(img)[:, :, l, c]
                    logits = F.log_softmax(log_prob, dim=1)
                    x = Categorical(logits=logits).sample([])
                    x = codebook(x)
                    for i in range(img.shape[0]):
                        img[i, :x.shape[1], l, c] = x[i]
        return img


class PixelCNN(PixCNNBase):
    def __init__(self, hid, sz):
        super(PixelCNN, self).__init__(3, hid, 256 * 3, sz)

    def forward(self, x):
        log_probs = super().forward(x)
        B, C, H, W = log_probs.shape
        return log_probs.view(B, 256, 3, H, W)

    def sample(self, temp, N):
        img = torch.zeros(N, 3, *self.sz, device=self.bias.device).uniform_(0, 1)
        return self.sample_(img, temp)


    def sample_cond(self, cond, temp):
        img = torch.empty(cond.shape[0], self.bias.shape[0], *self.sz,
                device=self.bias.device).uniform_(0, 1)
        cond_rsz = F.interpolate(cond, size=img.shape[2:], mode='nearest')
        img = torch.cat([img, cond_rsz], dim=1)
        return self.sample_(img, temp)[:, cond.shape[1]:]


    def sample_(self, img, temp=0):
        self.eval()
        with torch.no_grad():
            for l in range(self.sz[0]):
                for c in range(self.sz[0]):
                    log_prob = self(img * 2 - 1)[:, :, :, l, c] / temp
                    for i in range(img.shape[0]):
                        logits = F.log_softmax(log_prob[i].transpose(0, 1), dim=1)
                        x = Categorical(logits=logits).sample((1,))
                        img[i, :, l, c] = x.float() / 255
        return img


if __name__ == '__main__':
    img = torch.zeros(1, 1, 15, 15)
    conv = MaskedConv2d(1, 1, 5, center=True)
    conv.bias.data.zero_()
    conv.weight.data.fill_(1)
    img[0, 0, 7, 7] = 1

    for i in range(3):
        img = conv(img)
        #img.clamp_(min=0, max=1)
        plt.imshow(img[0, 0].detach().numpy())
        plt.show()

