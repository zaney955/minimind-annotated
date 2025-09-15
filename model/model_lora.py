import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A [rank, in_features], self.A(x)=xA^T
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B [out_features, rank]
        
        # 这种初始化保证 LoRA 在训练前不破坏原模型
        self.A.weight.data.normal_(mean=0.0, std=0.02)  # 矩阵A高斯初始化,学习起点随机小扰动
        self.B.weight.data.zero_()  # 矩阵B全0初始化,确保初始时LoRA不影响原模型输出

    def forward(self, x):
        return self.B(self.A(x))

# 给模型的所有线性层“外挂”一个 LoRA 模块
def apply_lora(model, rank=8):
    # 1.遍历模型的所有子模块
    for name, module in model.named_modules():  # 会递归列出模型里的所有子模块（例如 nn.Linear、nn.Conv2d 等）;name 是模块名，module 是实例对象
        # 2.只挑选方阵的 Linear 层
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:  # 条件 1：必须是 nn.Linear；条件 2：输出维度 = 输入维度（方阵）
            # 3.给这个 Linear 层加一个 LoRA 模块
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)  # 生成一个 LoRA 层，输入输出维度和这个 Linear 匹配
            setattr(module, "lora", lora)  # 把它作为属性挂到原来的 Linear 上，方便后续调用（比如 module.lora）
            
            # 4.保存原始的 forward：保留原本的 forward 方法（正常的 Linear 运算），等会要用它 + LoRA 的结果。
            original_forward = module.forward

            # 5.定义新的 forward，显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)  # 最终结果 = 原始权重 + LoRA 补丁

            # 6.替换原始 forward
            module.forward = forward_with_lora  # 把 Linear 的 forward 改写成 “原始 + LoRA”的形式；这样以后模型调用时，就会自动走带 LoRA 的逻辑。

# 只保存模型里的 LoRA 参数为一个新的权重文件
def save_lora(model, path):    
    state_dict = {}
    # 遍历模型所有子模块，只挑出带 lora 属性的模块
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            
            # 获取当前 LoRA 模块的 state_dict；把 key 前面加上模块名 + .lora. 前缀，避免不同层的 LoRA 参数重名：linear.lora.A.weight、linear.lora.B.weight,...
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            # 把这个 LoRA 模块的参数放到总的 state_dict 里
            state_dict.update(lora_state) 
    torch.save(state_dict, path)

# 从保存的 checkpoint 里提取对应层的 LoRA 参数，并加载到模型里
def load_lora(model, path):  # model: 待加载 LoRA 权重的模型实例；path:  LoRA 权重文件的路径
    state_dict = torch.load(path, map_location=model.device)
    # 遍历模型的所有子模块
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):  # 只挑出带 lora 属性的
            # 从整个 checkpoint 里筛选出属于当前层 LoRA 的参数
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}  # replace(f'{name}.lora.', '') 把前缀去掉，得到 LoRA 子模块能识别的 key
            # 把提取到的参数加载到这个 LoRA 模块里
            module.lora.load_state_dict(lora_state)


if __name__ == '__main__':
    # 测试：将 LoRA 应用到一个简单的线性模型上
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)  # 方阵线性层

        @property
        def device(self):
            return next(self.parameters()).device  # 返回模型参数所在设备

        def forward(self, x):
            return self.linear(x)


    model = TestModel()
    # 打印原始模型的结构
    print(model, "\n\n")
    
    # 注入LoRA：
    apply_lora(model, rank=4)
    print(model, "\n\n")
    
    for name, module in model.named_modules():
        print(name,':',module)