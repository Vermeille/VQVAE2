import functools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import SVHN, CIFAR100, CIFAR10
import torchvision.transforms as TF

import torchelie
from torchelie.recipes.trainandtest import TrainAndTest
from torchelie.distributions import Logistic, LogisticMixture
import torchelie.nn as tnn
from torchelie.optim import RAdamW


class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(*[
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1)
        ])

    def forward(self, x):
        return x + self.conv(x)


class SoftmaxOutput:
    output_ch = 256

    @staticmethod
    def loss(pred, y):
        y = (y * 255).long()
        out = torch.sum(
            nn.functional.softmax(pred, dim=1) *
            torch.linspace(0, 1, 256, device=pred.device).view(256, 1, 1, 1),
            dim=1)
        return nn.functional.cross_entropy(pred, y), out


class LogisticOutput:
    output_ch = 2

    @staticmethod
    def loss(pred, x):
        loc = pred[:, 0, ...]
        scale = pred[:, 1, ...].exp()
        p = Logistic(loc, scale)
        lp = p.log_prob(x)
        return -lp.mean(), loc


class LogisticMixtureOutput:
    n_mix = 10
    output_ch = 3 * n_mix

    @staticmethod
    def loss(pred, x):
        n_mix = LogisticMixtureOutput.n_mix

        weights = pred[:, 0:n_mix, ...]
        locs = pred[:, n_mix:n_mix * 2, ...]
        scales = pred[:, n_mix * 2:n_mix * 3, ...].exp()
        lm = LogisticMixture(weights, locs, scales, dim=1)

        return -lm.log_prob(x).mean(), lm.mean

class Classifier(nn.Module):
    def __init__(self,
                 vq_type,
                 vq_dim=32,
                 codes=512,
                 output_mode=LogisticMixtureOutput):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(3, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(256),
            ResBlock(256),
            nn.Conv2d(256, vq_dim, 1),
            self.get_vq(vq_dim, codes, vq_type),
            nn.Conv2d(vq_dim, 256, 1),
            ResBlock(256),
            ResBlock(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                256, 3 * output_mode.output_ch, 4, stride=2, padding=1)
        ])
        self.norm = tnn.ImageNetInputNorm()
        self.output_mode = output_mode
        self.vq_type = vq_type
        self.codes = codes

    def get_vq(self, num_channels, num_tokens, vq_type):
        if vq_type == 'none':
            return tnn.Dummy()
        if vq_type in ['angular', 'nearest']:
            return tnn.VQ(num_channels,
                          num_tokens,
                          dim=1,
                          return_indices=False,
                          mode=vq_type,
                          init_mode='first')

    def make_optimizer(self):
        return RAdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(self, x):
        out = self.model(self.norm(x))
        b, ch, h, w = out.shape
        return out.view(b, self.output_mode.output_ch, 3, h, w)

    def train_step(self, batch, opt):
        x, _ = batch
        opt.zero_grad()
        out = self(x)
        loss, out = self.output_mode.loss(out, x)
        loss.backward()
        opt.step()

        return {'loss': loss, 'metrics': {'out': out[:8].clamp(min=0, max=1)}}

    def validation_step(self, batch):
        x, _ = batch
        out = self(x)
        loss, out = self.output_mode.loss(out, x)
        return {'loss': loss}


ds = CIFAR10('~/.cache/torch/cifar10', download=True, transform=TF.ToTensor())
dt = CIFAR10('~/.cache/torch/cifar10',
             train=False,
             download=True,
             transform=TF.ToTensor())
dl = DataLoader(ds,
                batch_size=32,
                shuffle=True,
                pin_memory=True,
                num_workers=32)

dlt = DataLoader(dt,
                 batch_size=32,
                 shuffle=True,
                 pin_memory=True,
                 num_workers=32)

for vq_type in ['nearest', 'angular', 'none']:
    for dim in [64, 128, 256, 512]:
        for loss in [LogisticMixtureOutput, SoftmaxOutput]:
            print(vq_type, dim, loss.__name__)
            trainer = TrainAndTest(Classifier(vq_type,
                                              8,
                                              codes=dim,
                                              output_mode=loss),
                                   'ae-exps_{}_dim_{}_loss_{}'.format(
                                       vq_type, dim, loss.__name__),
                                   device='cuda')
            trainer(dl, dlt, epochs=20)
