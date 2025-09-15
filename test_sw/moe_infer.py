import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyExpert(nn.Module):
    """简单的专家模块"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 为了直观演示，不要随机权重，而是固定为单位矩阵
        nn.init.eye_(self.fc1.weight)
        nn.init.eye_(self.fc2.weight)

    def forward(self, x):
        # 专家就是恒等变换 + ReLU
        return F.relu(self.fc2(F.relu(self.fc1(x))))

class MiniMoE(nn.Module):
    def __init__(self, hidden_dim=4, n_experts=2, num_experts_per_tok=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.experts = nn.ModuleList([DummyExpert(hidden_dim) for _ in range(n_experts)])

    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount(minlength=self.n_experts).cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]].unsqueeze(1))
            expert_cache.scatter_add_(0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out)
        return expert_cache
    
    def moe_infer_simple(self, x, flat_expert_indices, flat_expert_weights):
        """
        x: [num_tokens, hidden_dim]
        flat_expert_indices: [num_tokens * num_experts_per_tok]
        flat_expert_weights: 同 shape
        """
        num_tokens = x.size(0)
        hidden_dim = x.size(1)
        output = torch.zeros_like(x)

        # 每个 token 处理 top_k 个专家
        num_experts_per_tok = self.num_experts_per_tok

        # 逐 token 处理
        for token_idx in range(num_tokens):
            # 当前 token 对应的专家索引和权重
            start = token_idx * num_experts_per_tok
            end = start + num_experts_per_tok
            experts_idx = flat_expert_indices[start:end]
            experts_w = flat_expert_weights[start:end]

            token_vec = x[token_idx]

            # 累加所有专家输出
            token_out = torch.zeros_like(token_vec)
            for e_idx, w in zip(experts_idx, experts_w):
                expert_out = self.experts[e_idx](token_vec.unsqueeze(0)).squeeze(0)
                token_out += w * expert_out

            output[token_idx] = token_out

        return output

# ===== 手动测试 =====
if __name__ == "__main__":
    # 输入：3个token，每个4维
    x = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],   # token0
        [5.0, 6.0, 7.0, 8.0],   # token1
        [9.0, 10.0, 11.0, 12.0] # token2
    ])

    model = MiniMoE(hidden_dim=4, n_experts=2, num_experts_per_tok=2)

    # 手动指定专家分配
    # 每个 token 走 top-2 个专家
    # flat_expert_indices shape = [num_tokens * top_k] = [3*2] = [6]
    flat_expert_indices = torch.tensor([0, 1,   # token0 -> expert0, expert1
                                        0, 1,   # token1 -> expert0, expert1
                                        0, 1])  # token2 -> expert0, expert1

    # 权重，简单点就直接指定
    flat_expert_weights = torch.tensor([0.6, 0.4,   # token0
                                        0.7, 0.3,   # token1
                                        0.2, 0.8])  # token2

    print("输入 x:\n", x)
    print("专家分配 flat_expert_indices:", flat_expert_indices)
    print("专家权重 flat_expert_weights:", flat_expert_weights)

    out = model.moe_infer(x, flat_expert_indices, flat_expert_weights)
    out2 = model.moe_infer_simple(x, flat_expert_indices, flat_expert_weights)
    
    print("输出 out:\n", out)
    print("输出 out2:\n", out2)
