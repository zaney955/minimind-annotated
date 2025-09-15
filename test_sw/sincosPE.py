import numpy as np
import matplotlib.pyplot as plt

# 正余弦位置编码函数
def positional_encoding(max_len, d_model, num_points=10):
    # 在每个整数位置之间增加采样点，让曲线更平滑
    pos = np.linspace(0, max_len-1, max_len*num_points)[:, np.newaxis]  # (max_len*num_points, 1)
    i = np.arange(d_model)[np.newaxis, :]    # (1, d_model)
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates

    # 偶数维度: sin，奇数维度: cos
    pos_encoding = np.zeros((pos.shape[0], d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pos_encoding, pos

# 参数
max_len = 110   # 序列长度
d_model = 6    # 维度数
num_points = 10 # 每个位置采样10个点，使曲线更平滑

# 生成编码
pos_encoding, pos = positional_encoding(max_len, d_model, num_points)

# 获取渐变颜色
colors = plt.cm.viridis(np.linspace(0, 1, d_model))

# 绘制曲线
plt.figure(figsize=(25, 6))
for dim in range(d_model):
    plt.plot(pos, pos_encoding[:, dim], color=colors[dim], label=f"dim {dim}")

plt.xlabel("Position")
plt.ylabel("Encoding Value")
plt.title("Sinusoidal Positional Encoding (Smooth Curves)")
plt.legend(loc="upper right", ncol=2, fontsize=8)
plt.grid(True)
plt.savefig("sincosPE.png")

plt.show()
