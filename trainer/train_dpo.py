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
from dataset.lm_dataset import DPODataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

# 把 模型输出的 logits 转换为 对应标签的对数概率
def logits_to_logprobs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    # Step 1: 计算每个 token 的 log-softmax 概率
    log_probs = F.log_softmax(logits, dim=2)  
    # log_probs: (batch_size, seq_len, vocab_size)

    # Step 2: 收集 labels 对应的 log 概率
    
    # torch.gather(..., dim=2): 从 log_probs 的第3维（vocab_size）中选择对应 label 的概率
    probs = torch.gather(log_probs, dim=2, 
                         index=labels.unsqueeze(2))  # labels.unsqueeze(2): (batch_size, seq_len, 1)
    # probs: (batch_size, seq_len, 1)

    probs = probs.squeeze(-1)   # probs: (batch_size, seq_len)  => 每个 token 的 log-probability
   
    return probs


def dpo_loss(ref_logprobs, logprobs, mask, beta):
    # ref_logprobs 和 logprobs 都是 shape: (batch_size, seq_len)
    # https://github.com/jingyaogong/minimind/issues/298
    """
    计算DPO (Direct Preference Optimization) 损失函数
    Args:
        ref_logprobs (torch.Tensor): 参考模型的对数概率，shape: (batch_size, seq_len)
        logprobs (torch.Tensor): 当前模型的对数概率，shape: (batch_size, seq_len)
        mask (torch.Tensor): 用于标记哪些 token 被计入损失（如生成部分），shape: (batch_size, seq_len)
        beta (float): DPO损失函数中的温度参数，控制优化强度
        
    Returns:
        torch.Tensor: 平均DPO损失值
    """
    # Step 1: 每个样本的有效长度（非 padding 部分 token 的数量）
    seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
    # Step 2: 对每个样本计算平均 log-probs，仅在 mask == 1 的位置有效
    ref_logprobs = (ref_logprobs * mask).sum(dim=1) / seq_lengths.squeeze()
    logprobs = (logprobs * mask).sum(dim=1) / seq_lengths.squeeze()

    # Step 3: 将 batch 划分为前一半为 chosen，后一半为 rejected
    batch_size = ref_logprobs.shape[0]
    
    chosen_ref_logprobs = ref_logprobs[:batch_size // 2]  # (batch_size // 2,)
    reject_ref_logprobs = ref_logprobs[batch_size // 2:]  # (batch_size // 2,)
    chosen_logprobs = logprobs[:batch_size // 2]  # (batch_size // 2,)
    reject_logprobs = logprobs[batch_size // 2:]  # (batch_size // 2,)

    # Step 4: log-ratio 比较（策略模型 vs 参考模型）  
    pi_logratios = chosen_logprobs - reject_logprobs  # (batch_size // 2,)
    ref_logratios = chosen_ref_logprobs - reject_ref_logprobs  # (batch_size // 2,)
    
    # Step 5: DPO 损失计算，鼓励 chosen 比 rejected 的分数更高
    # 参考模型通常是一个 未微调的语言模型 或 行为基线，它体现了“自然语言生成的默认概率分布”,DPO 不只是让模型更偏向chosen,而是 让模型比参考模型更偏向chosen
    logits = pi_logratios - ref_logratios  # (batch_size // 2,) 
    loss = -F.logsigmoid(beta * logits)  # (batch_size // 2,)  # sigmoid将logits映射到(0,1)区间；log将其映射到(-∞,0]区间；负号将其转为[0,+∞)区间，符合最小化目标。beta让sigmoid的曲线变化更平滑（控制优化强度）
    return loss.mean() # 标量，.mean()等价于DPO loss数学公式中的期望符号E


def train_epoch(epoch, wandb):
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        x_chosen = batch['x_chosen'].to(args.device)  # (batch_size, seq_len)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        
        x = torch.cat([x_chosen, x_rejected], dim=0)  # (2*batch_size, seq_len)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_logprobs = logits_to_logprobs(ref_logits, y)
            ref_logprobs = ref_logprobs * mask
            outputs = model(x)
            logits = outputs.logits
            logprobs = logits_to_logprobs(logits, y)
            logprobs = logprobs * mask
            loss = dpo_loss(ref_logprobs, logprobs, mask, beta=0.1)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/rlhf_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    # 初始化参考模型
    ref_model = MiniMindForCausalLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer


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
    parser = argparse.ArgumentParser(description="MiniMind RLHF")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    # sft阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-RLHF-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl")

    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Full-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

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

    model, ref_model, tokenizer = init_model(lm_config)

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
