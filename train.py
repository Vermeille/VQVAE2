import sys
import time
import copy

import torch
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF

import vqae
from perceptual_loss import PerceptualLoss
from patchgan import GANLoss

from visdom import Visdom

viz = Visdom()
viz.close()

SZ=128
tfms = TF.Compose([
        TF.Resize(SZ),
        TF.CenterCrop(SZ),
        TF.ToTensor(),
    ])

device = 'cuda'

class ForgivingDataset:
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        try:
            return self.ds[i]
        except Exception as e:
            print(e)
            if i < len(self):
                return self[i + 1]
            else:
                return self[0]

ds = torchvision.datasets.ImageFolder(sys.argv[1], tfms)
ds = ForgivingDataset(ds)
print('dataset size:', len(ds))
dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True,
        num_workers=16, pin_memory=True)
model = vqae.baseline(SZ, noise=True).to(device)
polyak = copy.deepcopy(model).eval()
#p_loss = PerceptualLoss(11).to(device)
ganloss = GANLoss().to(device)
opt = optim.Adam(model.parameters(), lr=2e-4)#, betas=(0.5, 0.99))

iters = 0
for epochs in range(30):
    for x, _ in dl:
        def go():
            global x
            x = x.to(device, non_blocking=True)
            opt.zero_grad()

            recon = model(x * 2 - 1)
            r_loss = F.l1_loss(recon, x)
            #loss = p_loss(recon, x)
            g_loss = ganloss(recon, x)
            loss = g_loss#r_loss# + 0.1 * g_loss
            loss.backward()
            opt.step()

            for pp, mp in zip(polyak.state_dict().values(), model.state_dict().values()):
                pp.data.copy_(pp.data * 0.998 + mp.data * 0.002)


            if iters % 10 == 0:
                #viz.line(X=[iters], Y=[r_loss.item()], update='append', win='r_loss', opts=dict(title='Reconstruction loss'))
                viz.line(X=[iters], Y=[g_loss.item()], update='append', win='g_loss', opts=dict(title='GAN loss'))
                with torch.no_grad():
                    recon = polyak(x * 2 - 1)
                viz.images(recon[:16].cpu().detach(), win='recon')
            if iters % 500 == 0:
                torch.save({'model': model.state_dict(), 'optim':
                    opt.state_dict(), 'loss': loss.item(), 'polyak':
                    polyak.state_dict()}, 'saved.pth')

        go()
        iters += 1
