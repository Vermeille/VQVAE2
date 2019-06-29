import sys
import time

import torch
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF

import vqae
import pixcnn
from perceptual_loss import PerceptualLoss
from vq import VQ

from visdom import Visdom

viz = Visdom()
viz.close()

SZ=64
tfms = TF.Compose([
        TF.Resize(SZ),
        TF.CenterCrop(SZ),
        TF.ToTensor(),
    ])

device = 'cuda'

ds = torchvision.datasets.ImageFolder(sys.argv[1], tfms)
#ds = torch.utils.data.Subset(ds, range(32))
dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
print('dataset size:', len(ds))

model_1 = pixcnn.PixelCNN(64, (SZ, SZ)).to(device)
opt_1 = optim.Adam(model_1.parameters(), lr=3e-2)

iters = 0
for epochs in range(30000):
    for x, _ in dl:
        def go():
            global x
            x = x.to(device)
            y = (x * 255).long()

            opt_1.zero_grad()
            z = model_1(x * 2 - 1)
            loss = F.cross_entropy(z, y)
            #loss += F.mse_loss(z[:, 1:], z[:, :-1])
            loss.backward()
            print(loss.item())
            opt_1.step()

            if iters % 10 == 0:
                print('SHOW')
                viz.line(X=[iters], Y=[loss.item()], update='append', win='loss')
                viz.line(X=torch.arange(256), Y=F.softmax(z[0, :, 0, 16, 16], dim=0).view(256).detach().cpu(), win='R')

            if iters % 500 == 0:
                with torch.no_grad():
                    model_1.eval()
                    img = model_1.sample(0.1, 4)
                    model_1.train()

                viz.images(img.cpu().detach(), win='recon')
                time.sleep(1)
            if iters % 100 == 0:
                pass#torch.save({'model_1': model.state_dict(), 'optim': opt.state_dict(), 'loss': loss.item()}, 'saved_pix_1.pth')

        go()
        iters += 1
