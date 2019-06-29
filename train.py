import sys
import time

import torch
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF

import vqae
from perceptual_loss import PerceptualLoss

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

ds = torchvision.datasets.ImageFolder(sys.argv[1], tfms)
print('dataset size:', len(ds))
dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)

model = vqae.baseline(SZ).to(device)
p_loss = PerceptualLoss(11).to(device)
opt = optim.Adam(model.parameters(), lr=3e-4)

iters = 0
for epochs in range(30):
    for x, _ in dl:
        def go():
            global x
            x = x.to(device)
            opt.zero_grad()

            recon = model(x * 2 - 1)
            loss = F.l1_loss(recon, x)
            #loss += p_loss(recon, x)
            loss.backward()
            opt.step()

            if iters % 10 == 0:
                viz.line(X=[iters], Y=[loss.item()], update='append', win='loss')
                viz.images(recon.cpu().detach(), win='recon')
                time.sleep(1)
            if iters % 100 == 0:
                torch.save({'model': model.state_dict(), 'optim':
                    opt.state_dict(), 'loss': loss.item()}, 'saved.pth')

        go()
        iters += 1
