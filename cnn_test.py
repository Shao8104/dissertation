from model import SimpleCNN
import torch

model = SimpleCNN()

x = torch.randn(1, 1, 28, 28)

output = model(x)

print(output.shape)