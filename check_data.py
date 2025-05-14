import torch

data = torch.load(f"data/M_D/data_{'train'}.pt")

print(data['xt'].shape)
print((data['yt'][0][:,None,:,:]) - 0.5)
print(data['param'].dtype)

