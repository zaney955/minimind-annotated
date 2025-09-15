import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm_origin(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps # 一个小的常数，用于防止除以零
        self.weight = nn.Parameter(torch.ones(dim)) # 可学习的缩放参数，维度为dim，初始值为 1

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

class RMSNorm_sw(torch.nn.Module):
    def __init__(self,dim: int,eps= 1e-5):
        super().__init__()
        self.eps = eps  # A small constant to prevent division by zero
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scaling parameter, dimension is dim, initial value is 1
        
    def _norm(self,x):
        return ((x*x).mean(-1,keepdim=True)+ self.eps).rsqrt()  # RMS normalization formula
    
    def forward(self,x):
        return x * self._norm(x) * self.weight
    
if __name__ == "__main__":
    # 实例化测试
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0]]])  # shape = (1, 2, 4)
    norm_origin = RMSNorm_origin(dim=4)
    output_origin = norm_origin(x)
    print(output_origin)# torch.Size([1, 2, 4])
    
    norm_sw = RMSNorm_sw(dim=4)
    output_sw = norm_sw(x)
    print(output_sw)# torch.Size([1, 2, 4])