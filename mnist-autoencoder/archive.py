
import torch


class AutoEncoder(torch.nn.Module):
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
