import torch

class CNNClassifier(torch.nn.Module):
    def __init__(self, n_input_channels=3, kernel_size=3, stride=2):
        super().__init__()
        layers=[32, 64, 120]
        L = [
          torch.nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=2, stride=stride),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        c = 16
        for l in layers:
          L.append(torch.nn.Conv2d(c, l, kernel_size=3, padding=2, stride=stride))
          L.append(torch.nn.ReLU())
          c = l
        L.append(torch.nn.Conv2d(c, 6, kernel_size=3, padding=2, stride=stride))
        self.layers = torch.nn.Sequential(*L)
        
    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.layers(x).mean([2,3])


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
