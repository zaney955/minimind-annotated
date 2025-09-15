import torch
from transformers import AutoTokenizer

# x = torch.tensor([[1]])  # shape = (2, 1)
# print("x:", x)

# y = x.expand(3, 3)  # 在列上扩展成 3
# print("y:", y)
# print("y shape:", y.shape)


# a = torch.tensor([1, 2])
# b = torch.tensor([3, 4, 5])
# o1 = torch.outer(a, b)
# o2 = torch.outer(b, a)

# print("o1:", o1)
# print("o1 shape:", o1.shape)
# print("--")
# print("o2:", o2)
# print("o2 shape:", o2.shape)


# a = torch.tensor([1,2,3,4,5])
# o=2**a
# print("o:", o)

# topk_idx = torch.tensor([[1, 3],
#                          [0, 2],
#                          [2, 4]])
# flat_topk_idx = topk_idx.view(-1)
# print("flat_topk_idx:", flat_topk_idx)


# flat_expert_indices = [0, 1, 1, 2, 0, 2, 0, 1]
# idxs = torch.tensor(flat_expert_indices).argsort()
# print("idxs:", idxs)


a = torch.tensor([1, 2, 3, 4, 5])
o = torch.tensor([1, 2, 0, 0, 0])
mask = a == o
tokenizer=AutoTokenizer.from_pretrained(r"./model/")
padding=tokenizer.pad_token_id
print("padding:", padding)
print("mask:", mask)
