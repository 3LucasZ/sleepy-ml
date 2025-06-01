import torch
f = open("dataset.txt")
dataset = f.read()
print("Data set length:", len(dataset))
chars = sorted(list(set(dataset)))
print("Characters found:", chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]


def decode(l): return ''.join([itos[i] for i in l])


data = torch.tensor(encode(dataset), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:50])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


torch.manual_seed(0)
batch_size = 4  # number of sequences processed in parallel
block_size = 8  # max content length for predictions


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
