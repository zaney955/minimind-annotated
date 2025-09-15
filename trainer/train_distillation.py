import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import math
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def distillation_loss_fn(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    # student_logits: 学生模型的原始logits (batch_size, seq_len, vocab_size)
    # teacher_logits: 教师模型的原始logits (batch_size, seq_len, vocab_size)

    # --- 1. 计算教师概率分布（soft targets） ---
    with torch.no_grad():
        # teacher_probs: (batch_size, seq_len, vocab_size)，每个位置的概率分布，softmax 后每行加和为1
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()  # .detach() → 避免反向传播进教师模型。

    # --- 2. 计算学生的 log softmax ---
    # student_log_probs: (batch_size, seq_len, vocab_size)，log 概率，用于 KL 散度计算
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # --- 3. 计算 KL 散度 ---
    # KL(P_teacher || P_student)：蒸馏损失，衡量教师分布和学生分布的差距
    # 用法有点“坑”，因为输入需要是 log 概率 和 概率，而不是两个概率分布直接丢进去
    kl = F.kl_div(
        student_log_probs,     # Input log-probabilities from student
        teacher_probs,         # Target probabilities from teacher
        reduction=reduction    # 默认 reduction='batchmean'：结果会除以 batch_size（而不是 token 总数）
    )

    # --- 4. 温度缩放补偿 ---
    # 因为 logits 除以了 temperature，所以梯度变小了，需要乘 T^2 来保持梯度 scale 一致
    return (temperature ** 2) * kl



def train_epoch(epoch, wandb, alpha=0.0, temperature=1.0):
    start_time = time.time()

    # 设置教师模型为 eval 模式，不参与梯度计算
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    # 遍历一个 epoch 中所有的 batch
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # X: [batch_size, seq_len]，模型输入
        # Y: [batch_size, seq_len]，ground truth label（一般是右移的 X）
        # loss_mask: [batch_size, seq_len]，标记有效位置

        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 获取当前 step 的学习率（可选 warmup/cosine）
        lr = get_lr(epoch * iter_per_epoch + step,
                    args.epochs * iter_per_epoch,
                    args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # === 学生模型前向传播 ===
        with ctx:  # 支持 mixed precision
            res = model(X)  # forward pass
            student_logits = res.logits  
            # student_logits: [batch_size, seq_len, vocab_size]

        # === 教师模型前向传播 ===
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X).logits
                # teacher_logits: [batch_size, seq_len, vocab_size_teacher]

                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]  # 如果学生词表比教师小，需要切片对齐，避免词表不一致

        # === Cross Entropy Loss（标准监督训练）（硬标签）===
        loss_mask_flat = loss_mask.view(-1)  
        # loss_mask_flat: [batch_size * seq_len]

        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),  # [batch_size * seq_len, vocab_size]
            Y.view(-1),  # [batch_size * seq_len]
            ignore_index=0,  # 忽略 padding token
            reduction='none'
        )  # ce_loss: [batch_size * seq_len]，逐 token loss
        
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        # mask 生效，仅统计有效 token 的平均 loss


        # 若模型为 MoE（Mixture of Experts），加上稀疏门控损失
        if lm_config_student.use_moe:
            ce_loss += res.aux_loss

        # === Distillation Loss（知识蒸馏）（软标签）===
        if teacher_model is not None:
            # 筛选有效位置再计算 KL 蒸馏 loss
            student_logits_flat = student_logits.view(-1, student_logits.size(-1))  # [batch_size * seq_len, vocab_size]
            teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))  # 同上

            distill_loss = distillation_loss_fn(
                student_logits_flat[loss_mask_flat == 1],  # [num_valid_tokens, vocab_size]
                teacher_logits_flat[loss_mask_flat == 1],  # [num_valid_tokens, vocab_size]
                temperature=temperature
            )  # 一般为 soft cross-entropy 或 KLDivLoss ;只在 mask=1 的 token 上计算 KD loss
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # === 总损失（加权融合）===
        # alpha 越大，越靠 ground-truth；越小，越依赖 teacher
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps
        # 注意：这里支持梯度累积（accumulation_steps > 1）

        # 反向传播（支持 AMP 自动混合精度）
        scaler.scale(loss).backward()

        # 每 accumulation_steps 次才执行一次优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 解除 AMP 放大
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            scaler.step(optimizer)  # 优化器更新
            scaler.update()  # AMP 更新 scaler
            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.4f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs - 1,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item(),
                    "ce_loss": ce_loss.item(),
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "last-time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/full_dist_{lm_config_student.hidden_size}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_student_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'学生模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)

    return model, tokenizer


def init_teacher_model(lm_config):
    model = MiniMindForCausalLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'教师模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_xxx.jsonl")

    args = parser.parse_args()
    # 定义学生模型和教师模型
    lm_config_student = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
    lm_config_teacher = MiniMindConfig(hidden_size=768, num_hidden_layers=16)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Dist-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

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

    # 初始化学生模型和教师模型
    model, tokenizer = init_student_model(lm_config_student)
    teacher_model = init_teacher_model(lm_config_teacher)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
