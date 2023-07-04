import torch

class Block(torch.nn.Module):
    def __init__(self, n_input, n_output, stride=1):
        super().__init__()
        self.net = torch.nn.Sequential(
          torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=1, bias=False),
          torch.nn.BatchNorm2d(n_output),
          torch.nn.ReLU(),
          torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.net2 = torch.nn.Sequential(
          torch.nn.BatchNorm2d(n_output),
          torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.net2(self.net(x) + x)


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        L = [
          torch.nn.Conv2d(3, 128, kernel_size=3, padding=1),
        ]
        
        for i in range(3):
          L.append(Block(128, 128))

        self.networks = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.networks(x).mean((2, 3))
        return self.classifier(x)


class FCN(torch.nn.Module):
    def __init__(self, n_input=3):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(n_input, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.up_conv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.up_conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.up_conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 5, kernel_size=4, stride=2 ,padding=1),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU()
        )
        self.ConvTran12 = torch.nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1 ,padding=1)
        self.ConvTran23 = torch.nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1 ,padding=1)
        self.ConvTran33 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1 ,padding=1)
        self.ConvTran32 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1 ,padding=1)
        self.ConvTran21 = torch.nn.ConvTranspose2d(3, 5, kernel_size=3, stride=1 ,padding=1)

    def forward(self, x):
        layer1 = self.conv1(x)
        layer2 = self.conv2(layer1) + self.ConvTran12(layer1)
        layer3 = self.conv3(layer2) + self.ConvTran23(layer2)
        up_layer3 = self.up_conv3(layer3) + self.ConvTran33(layer3)
        skip_layer = torch.cat([up_layer3, layer2], dim=1)
        up_layer2 = self.up_conv2(skip_layer) + self.ConvTran32(up_layer3)
        skip_layer = torch.cat([up_layer2, layer1], dim=1)
        up_layer1 = self.up_conv1(skip_layer) + self.ConvTran21(x)
        up_layer1 = up_layer1[:, :, :x.shape[2], :x.shape[3]]
        return up_layer1


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
