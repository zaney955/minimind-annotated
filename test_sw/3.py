import torch
from torch import nn

a = torch.tensor([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0]])

b=nn.Linear(3, 2, bias=False)  # [2,3]

# a- > out
o1=a@b.weight.T 
o2=b(a)
o3=(b.weight@a.T).T


# c=b(a)
# d=b(c)
print(o1)
print(o2)
print(o3)
print(b.bias)
print(b.weight)
# print(d)