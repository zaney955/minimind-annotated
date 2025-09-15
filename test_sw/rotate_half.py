import torch


# def rotate_half(x):
#     return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

# x= torch.tensor([0,1,2,3,4,5])
# print(rotate_half(x))


import torch

def rotate_half_A(x):
    # 定义A: (-x2, -x3, x0, x1)
    return torch.cat([-x[2:], x[:2]])

def rotate_half_B(x):
    # 定义B: (-x1, x0, -x3, x2)
    return torch.tensor([-x[1], x[0], -x[3], x[2]])

def rope(x, theta, rotate_half_fn):
    cos, sin = torch.cos(theta), torch.sin(theta)
    return x * cos + rotate_half_fn(x) * sin

# 随机四维向量
x = torch.tensor([1, 2, 3, 4])
theta = torch.tensor(0.7)  # 任意角度

res_A = rope(x, theta, rotate_half_A)
res_B = rope(x, theta, rotate_half_B)


# 验证两种旋转方式是否等价
print("输入向量 x:", x)
print("使用 rotate_half_A 结果:", res_A)
print("使用 rotate_half_B 结果:", res_B)
print("结果是否一致:", torch.allclose(res_A, res_B, atol=1e-6))
# print(res_A.multiply(res_B.mT))