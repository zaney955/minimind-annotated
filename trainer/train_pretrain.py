import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings("ignore")


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


# 余弦退火学习率
def get_lr(current_step, total_steps, lr):
    cos_lr = 0.5 * lr * (1 + 
                         math.cos(math.pi * current_step / total_steps) # 余弦退火公式: 初始时= cos(0) = 1→ 初始时 lr 接近 lr; 结束时= cos(pi) = -1 → 结束时 lr → 0
                         )
    min_lr = lr / 10 # 给 lr 加了一个 最低 lr（floor），避免 lr 降为
    return cos_lr + min_lr
# lr
# ^
# |.
# |      .
# |          .
# |             .
# |               .
# |                .
# |                 .
# |                   .
# |                      .
# |                          .
# |                                .
# +-----------------------------------> step


def train_epoch(epoch, wandb):
    # 定义交叉熵损失函数（不做reduction，保留每个token位置的loss）
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)  # [batch_size, seq_len]
        Y = Y.to(args.device)  # [batch_size, seq_len]
        loss_mask = loss_mask.to(args.device)  # [batch_size, seq_len]

        # 余弦退火学习率
        lr = get_lr(
            epoch * iter_per_epoch + step,  # 当前训练步数（全局step）
            args.epochs * iter_per_epoch,  # 总训练步数
            args.learning_rate,  # 初始学习率
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr # 动态更新优化器中的学习率

        with ctx:
            res = model(X)  # 前向传播，res.logits: [batch, seq_len, vocab_size]
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)),  # 转为2D: [batch*seq_len, vocab_size]
                            Y.view(-1)  # 展平目标: [batch*seq_len]
                            ).view(Y.size())
            
            loss = (loss * loss_mask).sum( # 总的有效 loss
                ) / loss_mask.sum()  # loss_mask.sum() 统计了所有有效 token 的数量
            loss += res.aux_loss  # MoE 模型的负载平衡 loss
            loss = loss / args.accumulation_steps  # 梯度累积：将loss缩小为1/N，以模拟N倍的 batch size

        scaler.scale(loss).backward()  # FP16 的表示范围小，梯度很容易变得太小（下溢）或者太大（溢出）：scaler.scale(loss) 会把 loss 放大一个系数（scale factor），保证梯度在 FP16 范围内.缩放后的梯度反向传播到参数上，防止梯度消失或 NaN

        if (step + 1) % args.accumulation_steps == 0:
            # 背景：AMP 训练时，梯度是先被 loss_scaling 放大了的,如果此时直接做 clip_grad_norm_，那算出来的梯度范数会被缩放过，不正确
            scaler.unscale_(optimizer) # 把放大的梯度 缩放回原始尺度，这样后续裁剪或正则化才是正确的
            # 对所有 model.parameters() 的 .grad 做梯度裁剪，避免梯度过大导致训练不稳定（梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # 具体做法：计算所有参数梯度的 L2 范数,如果超过 args.grad_clip，则整体缩小，使范数正好等于阈值
            
            scaler.step(optimizer) # 使用上面unscale后（原始未缩放）的梯度更新参数。1.检测梯度是否溢出（inf / nan），如果梯度正常：GradScaler 会先把梯度缩放回原来的尺度，再调用 optimizer.step() 更新参数;如果梯度溢出：本次更新被跳过，梯度不会破坏模型。2.动态调整 scale factor,根据是否溢出，下一轮 loss 会乘上更大或更小的 scale factor；保证 FP16 梯度既不下溢也不溢出)
            scaler.update() # 动态调整 scale factor（根据梯度是否溢出，自动调整 scale factor,保证下一轮训练梯度既不下溢也不溢出)

            optimizer.zero_grad(set_to_none=True) # set_to_none=True 会把梯度直接置为 None，比置0更节省内存（尤其在大模型上）官方文档和大多数例子都推荐 配合 AMP 时使用 set_to_none=True，因为：AMP 本来就更依赖内存优化。减少显存占用和无用计算。

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps, # 记录和判断收敛情况的loss：不要用 scaler 缩放后的 loss，同时乘以 args.accumulation_steps 恢复到原始值
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                )
            )

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log(
                    {
                        "loss": loss.item() * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]["lr"],
                        "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60
                        - spend_time // 60,
                    }
                )

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth"

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained("../model/")
    model = MiniMindForCausalLM(lm_config).to(args.device)
    Logger(
        f"LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )
    return model, tokenizer


def init_distributed_mode():
    if not ddp:
        return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 如果当前设备是 CPU，就没有必要用 AMP，于是用 nullcontext() 占位；如果是 GPU，就启用 autocast()，利用混合精度。
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    # torch.cuda.amp.GradScaler 用来动态缩放 loss，解决 半精度训练（FP16/BF16）梯度下溢或溢出 问题
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
