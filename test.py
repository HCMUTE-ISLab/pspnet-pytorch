from train import build_network
import torch


net , _= build_network(None, 'resnet18')

img = torch.rand(1,3,224,224)

print(net)
