# from kan import *

# model = KAN(width=[5,10,1], k = 10, grid = 5, seed = 1)
# x = torch.rand(100,2)
# y = torch.rand(100,1)
# dataset = create_dataset_from_data(x, y, device='cpu')

# print(dataset['train_input'].shape, dataset['train_label'].shape)
# model.fit(dataset, opt="LBFGS", steps=20, lamb=0.01);
# model.plot()

# x = torch.tensor([0.4, 0.3, 0.1, 0, 0])
# val = model.get_act(x)
# print(val)

from kan import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
# dataset = create_dataset(f, n_var=2, device=device)
x = torch.rand(100,4)
y = torch.rand(100,2)
dataset = create_dataset_from_data(x, y, device='cpu')
dataset['train_input'].shape, dataset['train_label'].shape

# train the model

out = 2
inp = 4
model = KAN(width=[inp,5,out], grid=5, k=3, seed=1, device=device)
model.fit(dataset, opt="LBFGS", steps=20, lamb=0.01);
model.plot()


x = torch.rand(1,4)
print(x)
y = model(x)
print(y)
print(y.max(1)[1].item())
print(y.max(1)[0].unsqueeze(1))