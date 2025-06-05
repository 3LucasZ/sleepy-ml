import os

import torch


dir_path = os.path.dirname(os.path.realpath(__file__))
model_filepath = os.path.join(dir_path, "model.pth")


def best_model_filepath(num):
    return os.path.join(dir_path, f"best_model_{num}.pth")


class AutoEncoder(torch.nn.Module):
    def __init__(self, latentDims):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(True),
            torch.nn.Linear(32, latentDims),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latentDims, 32),
            torch.nn.ReLU(True),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 28 * 28),
            torch.nn.ReLU(True),
            torch.nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec


class AutoEncoderCNN(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        latentDims = 10
        self.encoder = torch.nn.Sequential(
            # 1 x 28 x 28
            torch.nn.Conv2d(1, 4, kernel_size=5),
            torch.nn.ReLU(True),
            # 4 x 24 x 24
            torch.nn.Conv2d(4, 8, kernel_size=5),
            torch.nn.ReLU(True),
            # 8 x 20 x 20
            torch.nn.Flatten(),
            # 3200
            torch.nn.Linear(3200, latentDims),
            # 10
        )
        self.decoder = torch.nn.Sequential(
            # 10
            torch.nn.Linear(latentDims, 3200),
            torch.nn.ReLU(True),
            # 3200
            torch.nn.Unflatten(1, (8, 20, 20)),
            # 8 x 20 x 20
            torch.nn.ConvTranspose2d(8, 4, kernel_size=5),
            # 4 x 24 x 24
            torch.nn.ConvTranspose2d(4, 1, kernel_size=5),
            # 1 x 28 x 28
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
