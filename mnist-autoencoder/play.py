from matplotlib import pyplot as plt
import numpy as np
import torchvision
from utils import *

latentDims = 2
model = AutoEncoder(latentDims)
path = os.path.join(dir_path, "regular2dim", "model.pth")
buffer = torch.load(path, weights_only=True)
model.load_state_dict(buffer)
model.eval()

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform)

latent_values = []
labels = []
latent_values_by_digit = {i: [] for i in range(10)}

with torch.no_grad():
    for image, label in test_dataset:
        image = image.unsqueeze(0)
        latent = model.encoder(image)
        latent_np = latent.squeeze(0).numpy()
        latent_values.append(latent_np)
        labels.append(label)
        latent_values_by_digit[label].append(latent_np)

average_latents = {}
for digit in range(10):
    average_latents[digit] = np.mean(latent_values_by_digit[digit], axis=0)

latent_values = np.array(latent_values)
labels = np.array(labels)


def plot():
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latent_values[:, 0],  # x-coordinates
        latent_values[:, 1],  # y-coordinates
        c=labels,  # Color by class
        cmap='tab10',  # Use 10 distinct colors for digits 0-9
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('2D Latent Space of MNIST Test Dataset')
    plt.tight_layout()


def gen(digit):
    # input_tensors = torch.tensor([[300.0, 300.0]], dtype=torch.float)
    input_tensors = torch.tensor([average_latents[digit]])
    output_tensors = model.decoder(input_tensors)
    image = output_tensors[0].squeeze().detach().numpy()

    plt.imshow(image, cmap='gray')
    plt.tight_layout()


if __name__ == "__main__":
    genMode = False
    digit = 9
    if genMode:
        gen(digit)
    else:
        plot()
    plt.show()
