# https://developer.apple.com/metal/pytorch/
# https://docs.pytorch.org/docs/stable/notes/mps.html
# https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")
