from matplotlib import pyplot as plt
import torch
import torchvision
from utils import *

# Hyperparameters
learning_rate = 0.003
epochs = 40
batch_size = 64
latentDims = 10

# 1. convert to 1D tensor of pixels scaled to [0, 1]
# 2. normalize to mean, stddev of the pixels; found from https://github.com/pytorch/examples/blob/main/mnist/main.py
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_dataset))
print(len(test_dataset))


class AutoEncoder(torch.nn.Module):
    def __init__(self):
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


def train(model, train_loader, optimizer, epoch):
    model.train()
    print("Epoch:", epoch)
    for batch_idx, (imgs, _) in enumerate(train_loader):
        optimizer.zero_grad()
        reconstructed_imgs = model(imgs)
        loss = torch.nn.MSELoss(reduction='mean')(imgs, reconstructed_imgs)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, batch {batch_idx}, loss: {loss.item()}')


def test(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    num_batches = len(test_dataset)/batch_size
    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(test_loader):
            reconstructed_imgs = model(imgs)
            loss = torch.nn.MSELoss(reduction='mean')(imgs, reconstructed_imgs)
            test_loss += loss.item()
            if batch_idx == 0 and epoch % 2 == 0:
                plt.figure(figsize=(10, 8))
                for j in range(10):
                    plt.subplot(4, 5, j+1)
                    plt.imshow(imgs[j].squeeze().numpy(), cmap='gray')
                    plt.title("Original")
                    plt.axis('off')
                    plt.subplot(4, 5, j+11)
                    plt.imshow(
                        reconstructed_imgs[j].squeeze().numpy(), cmap='gray')
                    plt.title("Reconstructed")
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"epoch{epoch}.png")
                plt.close()
    test_loss /= num_batches
    print(f'Test batch average loss: {test_loss}')


def main():
    torch.manual_seed(32)
    model = AutoEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader, epoch)
    torch.save(model.state_dict(), model_filepath)


if __name__ == '__main__':
    main()
