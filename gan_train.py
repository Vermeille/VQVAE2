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

SZ=64
tfms = TF.Compose([
        TF.Resize(SZ),
        TF.CenterCrop(SZ),
        TF.ToTensor(),
    ])

device = 'cuda'

def forever(iterable):
    ii = iter(iterable)
    while True:
        try:
            yield next(ii)
        except Exception as e:
            ii = iter(iterable)

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


#ds = torchvision.datasets.MNIST('.', download=True, transform=tfms)
ds = torchvision.datasets.ImageFolder(sys.argv[1], tfms)
ds = ForgivingDataset(ds)
print('dataset size:', len(ds))
dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True,
        num_workers=16, pin_memory=True)

#p_loss = PerceptualLoss(11).to(device)

loss_fn = GANLoss(len(ds.ds.classes), addsign=loss_type=='gan-addsign',
        lamb=0).to(device)
model = vqae.baseline_128_n_1vq(len(ds.ds.classes)).to(device)

for m in model.modules():
    if hasattr(m, 'weight') and isinstance(m, (torch.nn.Conv2d)):
        pass#torch.nn.utils.spectral_norm(m)

polyak = copy.deepcopy(model).eval()
opt = optim.Adam(model.parameters(), lr=1e-5, betas=(0., 0.99))
if loss_type == 'gan-addsign':
    opt = AddSign(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

ACC=2
iters = 0
for epochs in range(100):
    for x, y in dl:
        def go():
            global x
            global y
            x = x.to(device, non_blocking=True)
            print(y)
            y = y.to(device, non_blocking=True)

            recon = model((x * 2 - 1, y))
            loss = loss_fn(recon, x, y, step=iters % ACC == 0)
            loss.backward()
            if iters % ACC == 0:#ACC // 2:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 50)
                print('update G')
                opt.step()
                opt.zero_grad()

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
            if iters % 500 == 0:
                torch.save({'model': model.state_dict(), 'optim':
                    opt.state_dict(), 'loss': loss.item(), 'polyak':
                    polyak.state_dict()}, '{}-{}-{}-{}.pth'.format(loss_type,
                        iters, sys.argv[1].split('/')[-1], tag))

        go()
        iters += 1
