import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 参数设置
# -------------------------
d_model = 8
seq_length = 50

# -------------------------
# 生成一个随机 embedding (模拟 token 表示)
# -------------------------
np.random.seed(42)
# embedding = np.random.randn(seq_length, d_model)
embedding = np.zeros((seq_length, d_model))


# -------------------------
# 正弦余弦位置编码 (Sinusoidal PE)
# -------------------------
PE = np.zeros((seq_length, d_model))
for pos in range(seq_length):
    for i in range(0, d_model, 2):
        div_term = 10000 ** (i / d_model)  # i 已经代表 2i
        PE[pos, i] = np.sin(pos / div_term)
        PE[pos, i+1] = np.cos(pos / div_term)
        
# 融合：embedding + PE
emb_sinusoidal = embedding + PE


# -------------------------
# 旋转位置编码 (RoPE)
# -------------------------
emb_rope = np.zeros_like(embedding)
for pos in range(seq_length):
    for i in range(0, d_model, 2):
        div_term = 10000 ** (i / d_model)
        theta = pos / div_term
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        x, y = embedding[pos, i], embedding[pos, i+1]
        emb_rope[pos, i]   = x * cos_t - y * sin_t
        emb_rope[pos, i+1] = x * sin_t + y * cos_t

# -------------------------
# 可视化对比
# -------------------------
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for i in range(d_model):
    plt.plot(range(seq_length), emb_sinusoidal[:, i], label=f'dim {i}')
plt.title("Sinusoidal Positional Encoding")
plt.xlabel("Position")
plt.ylabel("Value")
plt.legend(loc='upper right', fontsize='small', ncol=4)
plt.grid(True)

plt.subplot(1, 2, 2)
for i in range(d_model):
    plt.plot(range(seq_length), emb_rope[:, i], label=f'dim {i}')
plt.title("Rotary Positional Encoding (RoPE)")
plt.xlabel("Position")
plt.ylabel("Value")
plt.legend(loc='upper right', fontsize='small', ncol=4)
plt.grid(True)

plt.tight_layout()
plt.savefig("positional_encoding_comparison.png")
plt.show()

print("Comparison visualization complete.")
