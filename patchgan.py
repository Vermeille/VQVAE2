import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Norm(nn.Module):
    def __init__(self, ch):
        super(L1Norm, self).__init__()
        self.bias = nn.Parameter(torch.zeros(ch, 1, 1))
        self.weight = nn.Parameter(torch.zeros(ch, 1, 1))

    def forward(self, x):
        l1 = x.abs().sum(dim=(2, 3), keepdim=True)
        x = x - x.mean(dim=(2, 3), keepdim=True)
        x = x / l1
        return (1 + self.weight) * x + self.bias

class ResBlk(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, ks=3):
        super(ResBlk, self).__init__()
        self.out_ch = out_ch
        self.stride = stride
        self.go = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=ks // 2),
            nn.GroupNorm(32, out_ch),
            #L1Norm(out_ch),
            #nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        self.shortcut = None
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        out = self.go(x)

        if self.stride != 1:
            x = F.avg_pool2d(x, 2, self.stride, 1, ceil_mode=True)

        if self.shortcut is not None:
            x = self.shortcut(x)
        return out# + x

class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, 3),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2),
            self.layer(64, 256, 2, 4),
            self.layer(256, 256, 2, 4),
            self.layer(256, 512, 2, 4),
            self.layer(512, 512, 2, 4),
            self.layer(512, 512, 2, 4),
        ])

        self.pred = nn.Conv2d(512, 1, 1)

        self.embs = nn.Embedding(n_classes, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', a=0.2)
                nn.init.constant_(m.bias, 0)
                nn.utils.spectral_norm(m)
        #self.embs.weight.data.zero_()

        print('Discriminator has', sum(p.numel() for p in self.parameters()),
                'parameters')

    def layer(self, in_ch, out_ch, stride=2, ks=3):
        return ResBlk(in_ch, out_ch, stride, ks)

    def forward(self, x, y):
        acts = []
        for l in self.layers:
            x = l(x)
            acts.append(x)

        x_avg = F.adaptive_avg_pool2d(x, 1)[:, :, 0, 0]
        return self.pred(x)[:, :, 0, 0], acts# + torch.mm(x_avg, self.embs(y).t())[:, :, None, None], acts

class GANLoss(nn.Module):
    def __init__(self, n_classes, lamb=1000, addsign=False):
        super(GANLoss, self).__init__()
        self.d = Discriminator(n_classes)
        if addsign:
            self.opt = AddSign(self.d.parameters(), lr=1e-3, betas=(0.9, 0.99))
        else:
            self.opt = torch.optim.Adam(self.d.parameters(), lr=2e-4,
                    betas=(0., 0.999))
        self.lamb = lamb

    def freeze(self):
        self.d.eval()
        for p in self.d.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        self.d.train()
        for p in self.d.parameters():
            p.requires_grad_(True)

    def forward(self, x, y, cls, step):
        x_d = x.detach()

        assert x.shape[0] == y.shape[0]
        self.unfreeze()

        super_x = torch.cat([x_d, y], dim=0)
        super_cls = torch.cat([cls, cls], dim=0)

        super_pred, super_acts = self.d(super_x * 2 - 1, super_cls)

        pred_fake, pred_real = super_pred[:x.shape[0]], super_pred[x.shape[0]:]
        ref = super_acts[x.shape[0]:]

        loss_fake = torch.clamp(1 + pred_fake, min=0).mean()
        loss_real = torch.clamp(1 - pred_real, min=0).mean()
        loss = loss_fake + loss_real
        loss.backward()

        if step:
            torch.nn.utils.clip_grad.clip_grad_norm_(self.d.parameters(), 50)
            print('update D')
            self.opt.step()
            self.opt.zero_grad()

        self.freeze()

        pred, acts = self.d(x * 2 - 1, cls)
        percept = sum(F.l1_loss(xx, yy.detach()) for xx, yy in zip(acts, ref))
        return -pred.mean() + self.lamb * percept
