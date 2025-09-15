import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 正余弦位置编码
# -------------------------
def get_sincos_positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(1000.0)/d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE

# -------------------------
# RoPE
# -------------------------
def apply_rope(x, position):
    d_model = x.shape[0]
    x_ = x.reshape(-1,2)
    angle = position * np.exp(np.arange(0, d_model, 2) * -(np.log(1000.0)/d_model))
    cos_pos = np.cos(angle).reshape(-1,1)
    sin_pos = np.sin(angle).reshape(-1,1)
    x_rot = np.zeros_like(x_)
    x_rot[:,0] = x_[:,0]*cos_pos[:,0] - x_[:,1]*sin_pos[:,0]
    x_rot[:,1] = x_[:,0]*sin_pos[:,0] + x_[:,1]*cos_pos[:,0]
    return x_rot.reshape(-1)

def get_rope_encodings(seq_len, d_model, base_embedding):
    return np.array([apply_rope(base_embedding, pos) for pos in range(seq_len)])

# -------------------------
# 余弦相似度
# -------------------------
def cosine_similarity(v1, v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

# -------------------------
# 参数
# -------------------------
seq_len = 1000         # 更长序列
d_model = 128
np.random.seed(42)

# 随机 embedding 并归一化
base_embedding = np.random.randn(d_model)
base_embedding /= np.linalg.norm(base_embedding)

# -------------------------
# 生成编码
# -------------------------
PE_sin = get_sincos_positional_encoding(seq_len, d_model)
PE_rope = get_rope_encodings(seq_len, d_model, base_embedding)

# -------------------------
# 远程衰减曲线
# -------------------------
step = 10
positions = np.arange(0, seq_len, step)
sim_sin_decay = [cosine_similarity(PE_sin[0], PE_sin[i]) for i in positions]
sim_rope_decay = [cosine_similarity(PE_rope[0], PE_rope[i]) for i in positions]

# -------------------------
# 相对位置曲线
# -------------------------
max_distance = 1000
rel_positions = np.arange(0, max_distance, step)
sim_sin_rel_avg = [np.mean([cosine_similarity(PE_sin[i], PE_sin[i+dist]) 
                            for i in range(seq_len-dist)]) for dist in rel_positions]
sim_rope_rel_avg = [np.mean([cosine_similarity(PE_rope[i], PE_rope[i+dist]) 
                             for i in range(seq_len-dist)]) for dist in rel_positions]

# -------------------------
# 绘制双子图
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(16,6))

# 左图：远程衰减
axes[0].plot(positions, sim_sin_decay, marker='o', markersize=3, label="Sinusoidal PE")
axes[0].plot(positions, sim_rope_decay, marker='x', markersize=3, label="RoPE")
axes[0].set_xlabel("Position distance from pos=0")
axes[0].set_ylabel("Cosine similarity")
axes[0].set_title("Decay from Position 0")
axes[0].grid(True)
axes[0].legend()

# 右图：相对位置曲线
axes[1].plot(rel_positions, sim_sin_rel_avg, marker='^', markersize=3, label="Sinusoidal PE")
axes[1].plot(rel_positions, sim_rope_rel_avg, marker='s', markersize=3, label="RoPE")
axes[1].set_xlabel("Relative position distance")
axes[1].set_ylabel("Average cosine similarity")
axes[1].set_title("Average Relative Position Similarity")
axes[1].grid(True)
axes[1].legend()

plt.suptitle("Sinusoidal PE vs RoPE: Decay and Relative Position Comparison (Long Distance)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("pe_rope_combined_long_distance.png")
plt.show()
