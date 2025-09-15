import torch

# 目标张量 (4行5列) - 使用int64类型
self = torch.zeros(4, 5, dtype=torch.int64)
# 索引张量 (与src形状相同)
index = torch.tensor([[0, 1, 2, 3, 0],
                      [1, 2, 3, 0, 1]])
# 源张量 - 与目标张量保持相同类型（int64）
src = torch.tensor([[1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2]], dtype=torch.int64)

# 执行scatter_add_操作
self.scatter_add_(dim=0, index=index, src=src)

print(self)
