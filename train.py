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
from patchgan import GANLoss

from visdom import Visdom

from addsign import AddSign

loss_type = sys.argv[2]
tag = '' if len(sys.argv) < 3 else sys.argv[3]
viz = Visdom(env='VQVAE2:' + loss_type + ':' +sys.argv[1] + ':' + tag)
viz.close()

SZ=256
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
dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True,
        num_workers=16, pin_memory=True)

#p_loss = PerceptualLoss(11).to(device)

if loss_type == 'l1':
    loss_fn = (lambda recon, x, y: F.l1_loss(recon, x))
    model = vqae.baseline_256().to(device)
elif loss_type == 'l2':
    loss_fn = (lambda recon, x, y: F.mse_loss(recon, x))
    model = vqae.baseline_128().to(device)
elif loss_type == 'gan' or loss_type == 'gan-addsign':
    loss_fn = GANLoss(len(ds.ds.classes), addsign=loss_type=='gan-addsign',
            lamb=0).to(device)
    model = vqae.baseline_128_n_1vq(len(ds.ds.classes)).to(device)

    for m in model.modules():
        if hasattr(m, 'weight') and not isinstance(m, (VQ, torch.nn.Embedding)):
            torch.nn.utils.spectral_norm(m)
elif loss_type == 'perceptual':
    ploss = PerceptualLoss(7).to(device)
    model = vqae.baseline_256().to(device)
    loss_fn = (lambda recon, x, y: ploss(recon, x))

polyak = copy.deepcopy(model).eval()
opt = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

sched = optim.lr_scheduler.ReduceLROnPlateau(opt)

iters = 0
tot_loss = 0
loss_it = 0
for epochs in range(100):
    for x, y in dl:
        def go():
            global x
            global y
            global tot_loss
            global loss_it
            if iters % 1000 == 0 and loss_it != 0:
                print('Loss:', tot_loss /loss_it, 'lr:',
                    opt.param_groups[0]['lr'])
                sched.step(tot_loss / loss_it)
                tot_loss = 0
                loss_it = 0
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()

            recon = model((x * 2 - 1, y))
            loss = loss_fn(recon, x, y)
            loss.backward()
            tot_loss += loss.item()
            loss_it += 1
            opt.step()

            for pp, mp in zip(polyak.state_dict().values(), model.state_dict().values()):
                pp.data.copy_(pp.data * 0.998 + mp.data * 0.002)
            polyak.eval()


            if iters % 10 == 0:
                viz.images(recon[:16].cpu().detach(), win='recon',
                        opts={'title': 'recon'})
                #viz.line(X=[iters], Y=[r_loss.item()], update='append', win='r_loss', opts=dict(title='Reconstruction loss'))
                viz.line(X=[iters], Y=[loss.item()], update='append', win='g_loss', opts=dict(title='GAN loss'))
                with torch.no_grad():
                    recon = polyak((x[:16] * 2 - 1, y[:16]))
                viz.images(recon.cpu().detach(), win='polyak',
                        opts={'title': 'polyak'})
            if iters % 5000 == 0:
                torch.save({'model': model.state_dict(), 'optim':
                    opt.state_dict(), 'loss': loss.item(), 'polyak':
                    polyak.state_dict()}, '{}-{}-{}-{}.pth'.format(loss_type,
                        iters, sys.argv[1].split('/')[-1], tag))

        go()
        iters += 1
