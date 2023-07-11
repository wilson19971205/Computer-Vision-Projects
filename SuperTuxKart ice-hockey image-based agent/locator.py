import torch
import torch.nn.functional as F

def spatial_argmax(logit):
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)

class CNNRegressor(torch.nn.Module):
    def __init__(self, channels=[16, 32, 32, 32]):
        super().__init__()
        conv_block = lambda c, h: [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, c, 5, 2, 2), torch.nn.ReLU(True)]
        h, _conv = 3, []
        for c in channels:
            _conv += conv_block(c, h)
            h = c
        self._conv = torch.nn.Sequential(*_conv, torch.nn.Conv2d(h, 1, 1))

    def forward(self, img):
        x = self._conv(img)
        return spatial_argmax(x[:, 0])

def load_locator():
    from torch import load
    from os import path
    r = CNNRegressor()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'locator.th'), map_location='cpu'))
    return r