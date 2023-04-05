from torch import nn
import torch

# Hyper-parameters
nin = 4 + 3
nout = 4
hidl = 32
grid_dim = 16 # We assume square grid
batch_size = 1

# Define the conv network
# - Individual layers
l1 = nn.Conv2d(nin, hidl, kernel_size=3, stride=1, padding=1, bias=True)
l2 = nn.Conv2d(hidl, hidl, kernel_size=1, stride=1, padding=0, bias=True)
l3 = nn.Conv2d(hidl, nout, kernel_size=1, stride=1, padding=0, bias=True)

# - All together
layers = [l1, l2, l3]

# Forward pass
inp = torch.randn(batch_size, nin, grid_dim, grid_dim)
x = inp
for l in layers:
    x = l(x)

# Inspect the result
print(inp.shape)
print(x.shape)