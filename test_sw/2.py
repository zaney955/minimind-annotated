from torch import nn

embed_tokens = nn.Embedding(6400, 1024)
print(embed_tokens.weight.shape)  # torch.Size([6400, 1024])

print(embed_tokens.weight)