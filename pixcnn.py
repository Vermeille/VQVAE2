import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_chan, out_chan, ks, center):
        super(MaskedConv2d, self).__init__(in_chan, out_chan, ks, padding=ks // 2)
        self.mask = torch.ones(ks, ks)
        self.mask[ks // 2 + 1:, :] = 0
        self.mask[ks // 2, ks // 2 + (1 if center else 0):] = 0

    def forward(self, x):
        self.weight_orig = self.weight
        del self.weight
        self.weight = self.weight_orig * self.mask
        res = super(MaskedConv2d, self).forward(x)
        self.weight = self.weight_orig
        return res


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

