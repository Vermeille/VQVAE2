import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
import functools

class NetInputNorm(nn.Module):
    def __init__(self):
        super(NetInputNorm, self).__init__()
        self.register_buffer('norm_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('norm_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def __call__(self, input):
        return (input - self.norm_mean) / self.norm_std

class WithSavedActivations(nn.Module):
    def __init__(self, model):
        super(WithSavedActivations, self).__init__()
        self.model = model
        self.activations = {}
        self.detach = True

        # We want to save activations of convs and relus. Also, MaxPool creates
        # actifacts so we replace them with AvgPool that makes things a bit
        # cleaner.

        for name, layer in self.model.named_children():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(functools.partial(self._save, name))
            if isinstance(layer, nn.MaxPool2d):
                self.model[int(name)] = nn.AvgPool2d(2, 2)


    def _save(self, name, module, input, output):
        if self.detach:
            self.activations[name] = output.detach().clone()
        else:
            self.activations[name] = output.clone()

    def __call__(self, input, detach):
        self.detach = detach
        self.activations = {}
        self.model(input)
        acts = self.activations
        self.activations = {}
        return acts

    def to(self, device):
        self.model = self.model.to(device)


def PerceptualNet(l):
    m = M.vgg16(pretrained=True).eval()
    m = m.features[:l]
    print('LOSS IS')
    m = WithSavedActivations(m)
    print(m.model)
    return m

class PerceptualLoss(nn.Module):
    def __init__(self, l):
        super(PerceptualLoss, self).__init__()
        self.m = PerceptualNet(l)
        self.norm = NetInputNorm()

    def forward(self, x, y):
        ref = self.m(self.norm(F.interpolate(y, size=(224, 224), mode='nearest')), detach=True)
        acts = self.m(self.norm(F.interpolate(x, size=(224, 224), mode='nearest')), detach=False)
        loss = 0
        for k in acts.keys():
            loss += torch.nn.functional.mse_loss(acts[k], ref[k])
        return loss
