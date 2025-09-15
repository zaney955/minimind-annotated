# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps # 一个小的常数，用于防止除以零
        self.gamma = nn.Parameter(torch.ones(dim)) # weight.shape = (dim,)，即每个特征一个缩放参数，初始值为 1;最终的作用：在归一化之后，再为每个维度加一个 可学习的缩放；nn.Parameter：表示这是一个 可学习参数，训练时会更新

    def _norm(self, x):
        # 计算输入x在最后一个维度上的平方均值，加上eps后开平方取倒数（rsqrt）,用输入x乘以这个倒数，实现归一化
        return x * torch.rsqrt(  # 直接调用 rsqrt 比先 sqrt 再 1 / 更高效，尤其在 GPU 上
            x.pow(2).mean(-1, keepdim=True) 
            + self.eps) 

    def forward(self, x):
        return self.gamma * self._norm(x.float()).type_as(x)

# 计算RoPE（旋转位置编码）的cos、sin 频率
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算RoPE（旋转位置编码）的旋转频率
    Args:
        dim (int): 嵌入维度，即每个token的特征维度
        end (int, optional): 序列的最大长度，即你想要预计算多少个位置的 cos/sin, 默认为32k，相当于提前生成 32k 个位置的 cos/sin
        theta (float, optional): 频率计算的基数参数，默认为1e6,LLaMA 里就是这么设的
    """
    # 生成偶数索引 [0, 2, 4, ...]，因为 RoPE 通常对每两个维度一组进行旋转
    freqs = 1.0 / (
        theta ** (
            torch.arange(0, dim, 2
                         )[: (dim // 2)].float(
                             ) / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float() # shape = (end, dim//2)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    为 Q/K 向量应用旋转位置编码（Rotary Positional Embedding, RoPE）。

    ⚡ 特点：
    - 使用了旋转前置（pre-rotation）方法：通过 `rotate_half` 将向量的前后半段重新排列并取负，
      然后与 cos/sin 进行线性组合。
    - 虽然写法与标准 RoPE 公式略有不同，但数学结果与公式完全等价。
    
    参数：
    - q: torch.Tensor, shape = (batch, seq_len, dim)，Query 向量
    - k: torch.Tensor, shape = (batch, seq_len, dim)，Key 向量
    - cos: torch.Tensor, shape = (seq_len, dim)，预计算的 cos 矩阵
    - sin: torch.Tensor, shape = (seq_len, dim)，预计算的 sin 矩阵
    - position_ids: 可选位置索引（未使用）
    - unsqueeze_dim: 扩展 cos/sin 的维度，用于广播乘法
    
    返回：
    - q_embed, k_embed: 应用旋转位置编码后的 Q/K 向量，shape 与输入相同

    说明：
    RoPE 的核心思想是对每对 embedding 维度进行二维旋转，将位置信息编码进 Q/K。
    此实现通过前置旋转实现同样效果，便于计算和广播。
    """    
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

# 将键值对张量沿着注意力头的维度进行复制扩展。
# 在MQA、GQA中，键(key)和值(value)张量的头数少于查询(query)张量的头数，
# 需要通过复制扩展键和值张量来匹配查询张量的头数。
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: # 如果 n_rep == 1，则直接返回输入张量 x
        return x
    # 否则，在第4个维度插入一个新维度并扩展，然后重新reshape，实现键值头的重复复制
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    # Attention 初始化
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # 设置KV头数量(GQA,多个 Query 头共享同一个 KV 头)，如果未指定则与注意力头数量相同(MHA,标准多头注意力)
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0 # 确保注意力头数量能被KV头数量整除
        # 设置注意力头和KV头数量
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # 每个 KV head 被多少个 Q head 共享
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个注意力头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads # hidden_size表示每个token的维度，为每个注意力头拆分成多个小向量
        
        # 初始化QKV线性映射层
        # 将输入 token embedding 映射为 Q/K/V 向量
        # 注意: 通过线性层将整个 hidden_size 映射到每个 head 的 head_dim，而不是直接把 hidden_size 切成 num_heads 份。
        #       如果直接切分，每个头只能看到部分 hidden_size，表达能力会受限。
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)  # Q 总是投影成 完整的 num_heads
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # K/V 投影的 head 数取决于 num_key_value_heads
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # K/V 投影的 head 数取决于 num_key_value_heads
        
        # 初始化输出投影层，将多头注意力输出拼接回 hidden_size
        # 注意: 这里不是简单拼接，而是通过线性层进行可学习的混合，使每个 hidden_size 维度能整合来自不同头的信息。
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        # 初始化注意力dropout和残差dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor, # 输入张量，形状为(batch_size, seq_len, hidden_size)
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 旋转位置编码元组，包含(cos, sin)两个张量
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # 上一个时间步的key和value缓存，用于加速解码
                use_cache=False, # 是否使用缓存，用于生成时的kv_cache
                
                # 注意力掩码，用于屏蔽padding位置（padding的token不参与注意力计算）。输入序列: [我, 爱, 自然, 语言, <pad>, <pad>]attention_mask: [1, 1, 1, 1, 0, 0],值为 1 表示这个位置是有效 token，值为 0 表示这个位置是 padding，不应该参与注意力
                # 一个 batch 含多个长短不一的序列，所以要加padding填充到序列的最大长度（seq_len）
                attention_mask: Optional[torch.Tensor] = None): 
        bsz, seq_len, _ = x.shape
        # 投影 线性变换 -> Q, K, V: 线性层生成所有头,一次性把每个 token 映射到 所有 Q/K/V 头拼在一起的空间
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 再reshape拆成多个头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 旋转位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现, 将历史的 key/value 与当前拼接(训练时不需要)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        present_kv = (xk, xv) if use_cache else None

        # 重复 KV 头 & 转置: 让所有 Q head 对应到正确的 KV head
        # 在 GQA或 MQA 中，多个 Q 头共享同一个 KV 头,如果直接做 Q·K^T，会报错，因为头数不匹配,解决方案：repeat_kv作用：把少量 KV 头 重复 n_rep 次,让每个 Q 头都对应一个 KV 头
        xq, xk, xv = (
            xq.transpose(1, 2), # [batch, seq_len, num_heads, head_dim] → [batch, num_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep  # xk [batch, seq_len, num_key_value_heads, head_dim] → [batch, seq_len, num_key_value_heads*n_rep, head_dim] = [batch, seq_len, num_heads, head_dim] 
                      ).transpose(1, 2), # → [batch, num_heads, seq_len, head_dim]
            repeat_kv(xv, self.n_rep  # xv size 变化同上
                      ).transpose(1, 2) 
        )

        # 计算注意力
        if self.flash and seq_len != 1: # 使用 FlashAttention
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None
            # 使用 PyTorch 原生 flash attention（内置 causal）
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
            
        # 使用传统的注意力计算
        else: 
            # 注意力分数计算
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, num_heads, seq_len, seq_len]
            
            # 添加上三角 mask，实现 causal attention。scores+mask， 上三角变成-inf，经过softmax后趋于0，从而遮盖未来信息
            casual_mask=torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),  # 创建一个大小为 (seq_len, seq_len) 的矩阵，每个元素都是 -inf。
                diagonal=1 # 控制从哪条对角线开始保留上三角。在 causal mask 中使用 diagonal=1，表示屏蔽主对角线右上方的未来位置，保留当前 token 自己和之前的 token。
            ).unsqueeze(0).unsqueeze(0)  # unsqueeze：匹配 batch 和 head 维度
            scores = scores + casual_mask
            
            # attention 掩码（如 padding mask）
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [bsz,seq_len]-->[bsz,1,1,seq_len]。
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9 # attention_mask原来是 1 的地方（有效 token）变成 0；原来是 0 的地方（padding）变成 -1e9（非常小）
                # 将 scores 与 extended_attention_mask 相加：
                # - causal mask（scores） 已经把未来 token 的分数设为 -inf
                # - extended_attention_mask 把 padding token 的分数设为 -1e9
                # 这样经过 softmax 后，未来信息和 padding 信息的注意力权重都 ≈ 0，不被模型关注
                scores = scores + extended_attention_mask 
                
            # softmax + dropout
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # 注意力加权求和
            output = scores @ xv  # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)

        # 输出投影与残差 dropout # 还原输出形状，并通过输出线性层
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # -> transpose：(batch, seq_len, num_heads, head_dim) -> reshape：(batch, seq_len, hidden_size)
        output = self.resid_dropout(self.o_proj(output))
        return output, present_kv



class FeedForward(nn.Module):
    """
    前馈神经网络层（Feed Forward Network）
    
    该层是Transformer架构中的前馈网络组件，采用门控线性单元（GLU）结构，
    包含上投影、门控投影和下投影三个线性变换，以及激活函数和dropout。
    
    增强表达能力：门控机制可以动态选择哪些特征重要
    提升模型性能：比普通 FFN 更容易捕捉复杂模式
    训练更稳定：GEGLU/GLU 结构在大模型中比普通 FFN 表现更好
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 如果 intermediate_size（FFN 的中间维度） 没指定，就按 hidden_size * 8/3 来计算，然后做 64 对齐（常用 GPU 优化手段，保证维度是 64 的倍数）
        # 计算公式：先按hidden_size的8/3倍计算，然后向上取整到64的倍数。(“8/3” 其实是一个 经验公式，来自 Transformer 系列模型的设计经验，并不是严格的数学定律。它的作用是决定 FFN 的中间层维度（intermediate_size）相对于隐藏维度 hidden_size 的放大比例。)
        # 为什么选择 64? 1.向量化效率高：GPU 的 线程块 (warp) 一般是 32 个线程，很多底层库（如 cuBLAS、TensorRT）在做矩阵乘法时，会按 32 或 64 对齐块操作。2.减少空洞填充：如果矩阵维度不是 64 的倍数，GPU 会在计算时填充零，造成额外开销。3.训练速度更快：对齐到 64 的倍数可以最大化利用 GPU 的计算单元，使矩阵乘法效率最高。
        if config.intermediate_size is None:  # 中间层维度,如果没指定就按 8/3 倍 idden_size 算
            intermediate_size = int(config.hidden_size * 8 / 3)  # 同时确保中间层维度是64的整数倍
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 首先得到中间层
        self.act_fn = ACT2FN[config.hidden_act]                                               # 激活函数，信号开关
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)     # 主信息通道，再次映射到中间层
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)   # 压回原维度，把中间维度再压回 hidden_size，方便和后续模块（例如注意力）相加
        self.dropout = nn.Dropout(config.dropout)                                              # Dropout层，随机丢弃部分元素，提高泛化能力，防止过拟合

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    1.对每个 token 计算所有专家的概率。
    2.选择 top-k 专家并可归一化权重。
    3.提供辅助损失，保证专家负载均衡。
    4.输出 top-k 专家索引和权重，供 MoE 的后续专家层使用。
    """
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok # 每个 token 将被分配到 top-k 个专家
        self.n_routed_experts = config.n_routed_experts # 总可选专家数（不含共享专家）

        self.scoring_func = config.scoring_func # 评分函数（仅支持 softmax）
        self.alpha = config.aux_loss_alpha # 辅助损失权重系数（用于训练时平衡专家负载）
        self.seq_aux = config.seq_aux # 是否使用按序列维度的均衡损失（True/False）

        self.norm_topk_prob = config.norm_topk_prob # 是否对 top-k 权重归一化
        self.gating_dim = config.hidden_size # 门控输入 token 向量的维度（即 hidden_size）
        
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim))) # 门控网络的权重矩阵，形状 [n_routed_experts, hidden_size]
        self.reset_parameters() # 初始化权重

    # 初始化MoEGate 门控网络的权重
    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # 使用 Kaiming 均匀分布初始化权重矩阵，适合 ReLU 等非线性激活

    def forward(self, hidden_states):  # MoEGate 接收 hidden_states 为输入（即输入 token 的表示）
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)  # 形状重塑为 (batch_size * seq_len, hidden_size)，以便一次性处理所有 token
        logits = F.linear(hidden_states, self.weight, None)  # 对每个 token 和所有专家权重进行点积，得到每个 token 分配给每个专家的原始分数（logits） [batch_size*seq_len, n_routed_experts]
        
        # 1. 计算分数，对专家维度做 softmax，得到每个 token 对各个专家的概率
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)  # 对 logits 应用 softmax 函数，将分数转换为介于 0 到 1 之间的概率分布，代表每个 token 分配给每个专家的概率
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 2. 选 top-k 专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)  # topk_idx 是被选中的专家索引，topk_weight 是对应的值（prob）。[batch_size*seq_len, top_k]

        if self.top_k > 1 and self.norm_topk_prob: # 是否对 top-k 权重归一化
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20  
            topk_weight = topk_weight / denominator

        # 3. 仅训练阶段，计算辅助损失，鼓励专家均衡使用
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            # 3.1 按序列维度均衡
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
                
            # 3.2 全局辅助损失（非序列）
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss # topk_idx：每个 token 的 top-k 专家索引。topk_weight：对应的权重。aux_loss：辅助负载损失，可加入总损失函数中。


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = MoEGate(config)
        
        # 创建了一个由 n_routed_experts 个 FeedForward 网络组成的列表，并把它们注册成 nn.Module 子模块（创建路由专家列表，每个专家是一个前馈网络）
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts) # nn.ModuleList([FeedForward(config), FeedForward(config), FeedForward(config), ...共有 n_routed_experts 个])
        ])
        
        # 可选：共享专家网络（作用于每个 token）
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config) for _ in range(config.n_shared_experts)
            ])

    # token -> gate -> 选专家 -> expert 处理 -> 加权求和 -> token 输出
    def forward(self, x):
        """
        前向传播函数，实现混合专家模型（MoE）的计算逻辑
        参数:
        x (Tensor): 输入张量，形状为 [batch_size, seq_len, hidden_dim]
        返回:
        y (Tensor): 输出张量，形状与输入相同
        属性:
        - 通过门控机制动态选择专家子网络
        - 支持训练和推理两种计算模式
        - 包含辅助损失计算和共享专家分支
        """
        identity = x # 用于 residual 加上共享专家输出，[bsz, seq_len, hidden_size]
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # ===== 1. 门控阶段：选择 top-k 个专家 作为每个 token 的路由 =====
        # topk_idx: [bsz*seq_len, top_k]，每个 token (共bsz*seq_len个token) 选择的专家索引
        # topk_weight: [bsz*seq_len, top_k]，每个 token 选择的专家对应的权重（通常是 softmax 后得到的概率）
        # aux_loss: 负载均衡损失,让每个专家大致被分到 相同数量的 token，且分配的权重和均衡(鼓励模型均匀使用所有专家，防止部分专家过载)
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # ===== 2. Flatten token 维度，准备并行处理 token =====
        x = x.view(-1, x.shape[-1])  # 将输入 x 展开成二维张量 [bsz * seq_len, hidden_size]
        flat_topk_idx = topk_idx.view(-1)  # 将专家索引展平成一维 [bsz * seq_len * topk]
        
        # ===== 3.1. 专家工作，训练阶段 =====
        if self.training:
            # 每个 token 被复制 top_k 次，送入不同专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0) # 显式复制每个 token K 次：形状 [bsz * seq_len * topk, hidden]
            y = torch.empty_like(x, dtype=torch.float16) # 用于收集每个专家处理后的结果（半精度，减少显存；确保与后续 expert 输出 dtype 一致），
            
            # 遍历每个专家，让其处理分配给它的 token
            for i, expert in enumerate(self.experts):
                # 找出并输入所有分配给第 i 个专家的 token
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致 [bsz * seq_len * topk, hidden]
                
            # 每个 token 在多个专家的输出加权求和，得到 token 的最终前馈输出
            y = (
                y.view(*topk_weight.shape, -1) # 每个 token 的 K 个专家输出聚合在一起，方便按权重加权 [bsz*seq_len, top_k, hidden_size]
                * topk_weight.unsqueeze(-1) # 广播到 hidden 维度，保证每个专家的输出向量都乘上对应权重 [bsz*seq_len, top_k, 1]
                ).sum(dim=1) # 沿 K 维度求和 [bsz*seq_len, top_k, 1] -> [bsz*seq_len, hidden_size]
            y = y.view(*orig_shape) # 还原为 [bsz, seq_len, hidden_size]
            
        # ===== 3.2. 专家工作，推理阶段 =====
        else:
            y = self.moe_infer(x,  # [bsz * seq_len, hidden_size]
                               flat_topk_idx,  # [bsz * seq_len * topk]
                               topk_weight.view(-1, 1) # [bsz * seq_len, top_k] --> [bsz * seq_len * top_k, 1]
                               ).view(*orig_shape)
            # 最后y：[bsz, seq_len, hidden_size]
            
        # ===== 4. 可选：添加共享专家的输出 =====
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad() # 关闭梯度计算（推理路径不会更新参数）
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理阶段的 MoE 前向传播。按照专家编号将 token 分组，分别送入对应专家中计算后合并。
        
        参数：
        - x: [bsz * seq_len, hidden_size]             所有 token 的表示（没有复制）
        - flat_expert_indices: [bsz * seq_len * top_k]  每个 token 被路由到的专家编号
        - flat_expert_weights: [bsz * seq_len * top_k , 1] 每个专家对应的门控权重
        """
        
        # 初始化输出缓存，与 x 同 shape 和 dtype 
        expert_cache = torch.zeros_like(x) # [bsz * seq_len, hidden_size]
        # 1. 根据专家编号对所有 token 排序（为了把分配到相同专家的 token 放到一起）
        idxs = flat_expert_indices.argsort()
        # 2. 统计每个专家分配到的 token 数量并累加，方便切分
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0) # 例FEI = [0, 1, 0, 2, 1, 3]-> .bincount() = [2, 2, 1, 1]-> .cumsum(0) = [2, 4, 5, 6]。则专家0的 token 在排序后的 flat 索引范围[0:2],专家1[2:4],专家2[4:5],专家3[5:6]
        # 3. 计算按照专家分组排序后的 token 属于哪些原始 token（因为每个 token 会被复制 top_k 次）
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # 4. 遍历每个专家，将分配到该专家的 token 送入对应 FFN 计算
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i] # 取出第 i 个专家 FFN
            # 5. 获取分配给第 i 个专家的 token 原始位置索引
            exp_token_idx = token_idxs[start_idx:end_idx] # 该专家负责的一组 token 索引
            # 6. 取出对应 token 表示
            expert_tokens = x[exp_token_idx] # 取出该专家对应 token 的输入切片：形状 [num_token_i, hidden_size]
            # 7. 执行当前专家对应 FFN 的前向传播，并转成缓存的 dtype
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 8. 用 gate 权重缩放输出
            # [num_token_i, hidden_size] * [num_token_i, 1] --> [num_token_i, hidden_size]
            expert_out.mul_(
                flat_expert_weights[idxs[start_idx:end_idx]] # 取出当前专家处理的 token对应的权重
                ) # MoE 的输出公式是加权求和。在推理时，每个专家只处理自己分配到的 token，所以先把 expert_out 乘以对应权重，后面 scatter_add_ 累加到最终输出就不需要再重复乘权重了
            
            # 9. 累加到输出缓存中，支持一个 token 被多个专家处理后结果叠加(每个 token 会被送到若干个专家, 专家输出结果后，需要“路由”回 token 的原始位置，并且加权)
            expert_cache.scatter_add_(
                0, # dim
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), # 存放“这个专家输出对应回哪个 token”
                expert_out # 专家的输出结果（已乘以权重）
                )

        return expert_cache


# 对应MiniMind架构图中的Transformer Layer k
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        # 基础模块
        self.num_attention_heads = config.num_attention_heads  # 8
        self.hidden_size = config.hidden_size  # 512
        self.head_dim = config.hidden_size // config.num_attention_heads  # 64
        
        # 自注意力模块，内部实现了 RoPE 相对位置编码
        self.self_attn = Attention(config)

        # 当前 Block 的层编号（可用于层内权重共享、分层控制等）
        self.layer_id = layer_id
        
        # Attention 前的 RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Attention 后 FFN 前 RMSNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 前馈网络模块，可配置是否使用专家混合（MoE）
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, 
                hidden_states,  # 输入的隐藏状态 [batch_size, seq_len, hidden_dim]
                position_embeddings,  # RoPE 位置编码 [seq_len, head_dim]
                past_key_value=None,  # KV 缓存 (key, value)，用于加速推理
                use_cache=False,  # 是否缓存当前层的 KV
                attention_mask=None  # attention 掩码
                ):
        # ---------------------- Self-Attention 层 ----------------------
        residual = hidden_states  # 保存残差连接
        
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # RMSNorm
            position_embeddings,  # Rotary PE 传入 Attention
            past_key_value,  # 上一个时间步的key和value缓存，用于加速解码
            use_cache,  # 是否缓存当前层 KV（一般在推理阶段使用）
            attention_mask  # 注意力掩码（padding token不计算注意力矩阵）
        )
        
        # 残差连接：原始输入 + attention 输出
        hidden_states += residual
        
        # ---------------------- MLP 层 ----------------------
        # MLP 前再做一次 RMSNorm
        hidden_states = hidden_states 
        + self.mlp(self.post_attention_layernorm(hidden_states))  # MLP 前再做一次 RMSNorm
        
        # 返回新的 hidden_states 和 当前层的 KV 缓存
        return hidden_states, present_key_value

# 整个 Transformer 主干网络，由多层 MiniMindBlock 堆叠而成
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # 用于将 token id 映射为可训练的向量, 初始化默认均匀分布, [vocab_size, hidden_size]。
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])  # 构建 num_hidden_layers 个 Transformer Block 层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # Transformer 后的 LayerNorm 层

        # 预计算 RoPE 所需的位置频率向量 (cos/sin)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        # 注册为 buffer（模型中持久存储但不参与优化）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 注册为 buffer（模型中持久存储但不参与优化）
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)  # 注册为 buffer（模型中持久存储但不参与优化）

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,  # 输入序列，[batch_size, seq_len]
                attention_mask: Optional[torch.Tensor] = None,  # padding mask [batch_size, seq_len]
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,  # 缓存 过去的 KV [(k_cache, v_cache),...]，长度等于层数=num_hidden_layers
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        # past_key_values = past_key_values or [None] * len(self.layers)
        if past_key_values is None or past_key_values == []:  # 如果没传入缓存，初始化为空（推理时才使用KV缓存）
            past_key_values = [None] * len(self.layers)  # [None, None, ..., None]，长度等于层数=num_hidden_layers
        
        # 为每个新生成的 token 计算正确的位置编码起始位置
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  # past_seq_len

        hidden_states = self.dropout(
            self.embed_tokens(input_ids)  # 注意这不是矩阵运算，而是查表操作：例如 input_ids=[[1,3]]，则 embed_tokens(input_ids)=[embed_tokens第1行, embed_tokens第3行]
            )  

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []  # 存储 KV cache：每层一个 (key, value)
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        # 如果使用了 MOE（稀疏专家），则合并辅助损失
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


# 面向因果语言建模的接口封装（Hugging Face 风格）
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):  # 继承了 GenerationMixin，其里面实现了 自回归生成逻辑（预测 → 拼接 → 再预测）
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        
        # 模型主干：MiniMindModel，输出 hidden_states
        self.model = MiniMindModel(self.config)

        # 输出层：将 hidden_size 映射为 vocab_size（即每个 token 的 logits）
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # [vocab_size, hidden_size]

        # 权重绑定：embedding 权重与 lm_head 权重共享
        self.model.embed_tokens.weight = self.lm_head.weight  # 共享权重，减少参数量 [vocab_size, hidden_size]

        # 输出容器（Hugging Face Transformers 定义的标准输出对象）
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
                attention_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,  # 控制 logits 保留哪些 token，一般训练时设置为0，推理时设置为1
                **args):
        
        # 调用主干模型，输出 hidden_states、presents（KV缓存）、aux_loss
        h, present_kvs, aux_loss = self.model(
            input_ids=input_ids,                                            # 输入 token 序列
            attention_mask=attention_mask,                                  # 用于 mask padding 的 attention mask
            past_key_values=past_key_values,                                # 用于增量推理的 KV 缓存
            use_cache=use_cache,                                            # 是否返回 KV cache
            **args
        )
        
        # 根据 logits_to_keep 参数决定保留输出的哪些位置
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        if isinstance(logits_to_keep, int):  # 当 logits_to_keep 是整数时
            # 取序列最后 logits_to_keep 个 token
            slice_indices = slice(-logits_to_keep, None)  # -logits_to_keep 表示 从序列倒数第 logits_to_keep 个 token 开始; None 表示一直到序列末尾
        else:  # 如果 logits_to_keep 不是整数，而是一个 tensor 或索引列表，就直接用它做切片
            slice_indices = logits_to_keep
        
        # 从 h 中保留最后 logits_to_keep 个token，送入 lm_head 做分类
        logits = self.lm_head(h[:, slice_indices, :]) # 训练时，slice_indices 是 0，logits 相当于 self.lm_head(h[:, 0:, :])，即整个 h;推理是1，logits 相当于 self.lm_head(h[:, -1:, :])，即最后一个 token 的 hidden_state
        # logits.shape: [batch_size, logits_to_keep, vocab_size]        self.OUT.__setitem__('last_hidden_state', h)
        
        # 构建结构化输出字典
        self.OUT.__setitem__('last_hidden_state', h)          # [batch_size, seq_len, hidden_size]
        self.OUT.__setitem__('logits', logits)                # [batch_size, logits_to_keep, vocab_size]
        self.OUT.__setitem__('aux_loss', aux_loss)            # scalar or tensor
        self.OUT.__setitem__('present_key_values', present_kvs)     # list of tuples: (key, value)
        return self.OUT
