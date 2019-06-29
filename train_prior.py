import sys
import time
#import tqdm

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
print('dataset size:', len(ds))
ds = torch.utils.data.Subset(ds, range(4096 * 2))
dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

get_vq = vqae.baseline(SZ)
get_vq.load_state_dict(torch.load('saved_64.pth')['model'])
get_vq = get_vq.to(device).eval()
encoder = get_vq[0]
decoder = get_vq[1]

codebooks = [m.embedding for m in encoder.modules() if isinstance(m, VQ)]

print('Encoding dataset...')
codes_1, codes_2 = None, None
idxs_1, idxs_2 = None, None
with torch.no_grad():
    for x, _ in dl:#tqdm.tqdm(dl, total=len(dl)):
        x = x.to(device)
        qs, idx = encoder(x * 2 - 1, True)

        qs = [qs[0].cpu(), qs[1].cpu()]
        idx = [idx[0].cpu(), idx[1].cpu()]

        if codes_1 is None:
            codes_1 = qs[0]
            codes_2 = qs[1]
            idxs_1 = idx[0]
            idxs_2 = idx[1]
        else:
            codes_1 = torch.cat([codes_1, qs[0]], dim=0)
            codes_2 = torch.cat([codes_2, qs[1]], dim=0)
            idxs_1 = torch.cat([idxs_1, idx[0]], dim=0)
            idxs_2 = torch.cat([idxs_2, idx[1]], dim=0)
        time.sleep(0.1)


del encoder
print('done')

model_1 = pixcnn.CodebookCNN(64, 64, 128, (16, 16)).to(device)
model_2 = pixcnn.CodebookCNN(128, 64, 128, (32, 32)).to(device)
opt_1 = optim.Adam(model_1.parameters(), lr=3e-2)
opt_2 = optim.Adam(model_2.parameters(), lr=3e-2)

dl = torch.utils.data.DataLoader(list(zip(codes_2, idxs_2, codes_1, idxs_1)),
        batch_size=32, shuffle=True)
iters = 0
for epochs in range(30000):
    for (x, y, x2, y2) in dl:
        def go():
            global x
            global y
            global x2
            global y2
            x = x.to(device)
            y = y.to(device).squeeze(1)

            x2 = x2.to(device)
            y2 = y2.to(device).squeeze(1)

            opt_1.zero_grad()
            z = model_1(x)
            loss_1 = F.cross_entropy(z, y)
            loss_1.backward()
            opt_1.step()

            opt_2.zero_grad()
            z = model_2(torch.cat([x2, F.interpolate(x, size=x2.shape[2:])], dim=1))
            loss_2 = F.cross_entropy(z, y2)
            loss_2.backward()
            opt_2.step()

            if iters % 10 == 0:
                viz.line(X=[iters], Y=[loss_1.item()], update='append',
                        win='loss1', opts=dict(title='loss 1'))
                viz.line(X=[iters], Y=[loss_2.item()], update='append',
                        win='loss2', opts=dict(title='loss 2'))
            if iters % 100 == 0:
                with torch.no_grad():
                    model_1.eval()
                    lat = model_1.sample(codebooks[1], 0.01, 4)
                    model_1.train()

                    model_2.eval()
                    lat_2 = model_2.sample_cond(lat, codebooks[0], 0.1)
                    model_2.train()
                    img = decoder([lat_2, lat])
                viz.images(img.cpu().detach(), win='recon')
                time.sleep(1)
            if iters % 100 == 0:
                pass#torch.save({'model_1': model.state_dict(), 'optim': opt.state_dict(), 'loss': loss.item()}, 'saved_pix_1.pth')

        go()
        iters += 1
