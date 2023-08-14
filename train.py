import sys
import time
import copy

import torch
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF

import vqae
from torchelie.loss import PerceptualLoss
from torchelie.datasets import NoexceptDataset
from torchelie.recipes.trainandtest import TrainAndTest
from torchelie.optim import RAdamW, Lookahead
import torchelie.nn as tnn
from patchgan import GANLoss

from visdom import Visdom

loss_type = sys.argv[2]
tag = '' if len(sys.argv) < 3 else sys.argv[3]

SZ = 128
tfms = TF.Compose([
    TF.Resize(SZ),
    TF.CenterCrop(SZ),
    TF.ToTensor(),
])

if loss_type == 'l1':
    loss_fn = (lambda recon, x, y: F.l1_loss(recon, x))
    model = vqae.baseline_256()
elif loss_type == 'l2':
    loss_fn = (lambda recon, x, y: F.mse_loss(recon, x))
    model = vqae.baseline_256()
elif loss_type == 'perceptual':
    ploss = PerceptualLoss(['conv1_2', 'conv2_2', 'conv3_2'])
    model = vqae.baseline_256()
    loss_fn = (lambda recon, x, y: ploss(recon, x))


class Classifier(torch.nn.Module):
    def __init__(self, model, loss):
        super(Classifier, self).__init__()
        self.model = model
        self.norm = tnn.ImageNetInputNorm()
        self.loss = loss

    def make_optimizer(self):
        return Lookahead(RAdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5))

    def forward(self, x):
        return self.model(self.norm(x))

    def train_step(self, batch, opt):
        x, _ = batch
        opt.zero_grad()
        out = self(x)
        loss = self.loss(out, x)
        loss.backward()
        opt.step()

        return {'loss': loss, 'metrics': {'out': out[:8].clamp(min=0, max=1)}}

    def validation_step(self, batch):
        x, _ = batch
        out = self(x)
        loss = self.loss(out, x)
        return {'loss': loss}


trainer = TrainAndTest(Classifier(model, F.mse_loss),
                       visdom_env=tag,
                       device='cuda')

#ds = torchvision.datasets.ImageFolder(sys.argv[1], tfms)
#ds = NoexceptDataset(ds)
ds = torchvision.datasets.CelebA('~/.cache/torch/celeba', download=True, transform=tfms)
print('dataset size:', len(ds))
dl = torch.utils.data.DataLoader(ds,
                                 batch_size=16,
                                 shuffle=True,
                                 num_workers=16,
                                 pin_memory=True)
dlt = torch.utils.data.DataLoader(torch.utils.data.Subset(
    ds, list(range(200))),
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=16,
                                  pin_memory=True)

trainer(dl, dlt, 100)
