import math
import matplotlib.pyplot as plt

# 定义学习率调度函数
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

# 超参数
total_steps = 1000      # 总训练步数
lr_init = 1e-3          # 初始学习率

# 计算每个 step 的学习率
lr_list = [get_lr(step, total_steps, lr_init) for step in range(total_steps)]

# 绘图
plt.figure(figsize=(8,4))
plt.plot(range(total_steps), lr_list, label='Cosine LR with floor')
plt.xlabel('Training step')
plt.ylabel('Learning rate')
plt.title('Cosine Annealing Learning Rate Schedule')
plt.grid(True)
plt.legend()
plt.savefig("cosine_annealing.png")
plt.show()
