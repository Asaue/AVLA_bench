# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator
from datasets import create_dataloader
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor
from peft import LoraConfig, get_peft_model



import logging
import os
import sys

import requests
import threading
import numpy as np

import requests
import threading
import numpy as np

import requests
import base64
from PIL import Image
import io
import time
import torch

def run_closed_loop_rollout(model, processor, batch_inputs, max_steps=30):
    print(f"\n🎮 [闭环测试] 暂停训练，开始 {max_steps} 步实机连贯操作...")
    
    # 兼容多卡
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()

    # 🚨 1. 瞬间重置仿真环境
    # try:
    #     requests.get("http://127.0.0.1:9380/reset", timeout=1.0)
    #     time.sleep(0.5) 
    # except Exception as e:
    #     print(f"⚠️ 无法连接到 MuJoCo 仿真器: {e}，跳过本次测试。")
    #     base_model.train()
    #     return

    device = batch_inputs["input_ids"].device

    # =========================================================
    # 🌟 核心修复：直接从当前合法的 Batch 中“借用”静态上下文数据
    # 绝对保证 domain_id, proprio, input_ids 的维度和索引百分百合法！
    # =========================================================
    safe_input_ids = batch_inputs["input_ids"][0:1]  # 借用合法的文本 Token
    safe_domain_id = batch_inputs["domain_id"][0:1]  # 借用合法的 domain
    safe_proprio = batch_inputs["proprio"][0:1]      # 借用合法的 proprio 初始状态

    for step in range(max_steps):
        try:
            # --- A. 获取真实的实时图片 ---
            resp = requests.get("http://127.0.0.1:9380/obs", timeout=1.0).json()
            img_global_bytes = base64.b64decode(resp["image_global_b64"])
            img_global = Image.open(io.BytesIO(img_global_bytes)).convert("RGB")
            
            # --- B. 仅处理视觉输入 ---
            # 传一个空文本进去，我们只想要它吐出的 image_input 和 image_mask
            live_vision = processor([img_global], "dummy text") 

            # --- C. 组装终极混合 Input ---
            inference_inputs = {
                "input_ids": safe_input_ids,
                "image_input": live_vision["image_input"].to(device),
                "image_mask": live_vision["image_mask"].to(device),
                "domain_id": safe_domain_id,
                "proprio": safe_proprio
            }

            # --- D. 模型推理 ---
            with torch.no_grad():
                pred_actions = base_model.generate_actions(**inference_inputs, steps=10)
            
            # 切割 10 维并发送
            raw_20d_action = pred_actions[0, 0].cpu().numpy()
            action_10d = raw_20d_action[:10].tolist()

            requests.post("http://127.0.0.1:9380/action", json={
                "action_type": "ee_pos",
                "data": action_10d,
                "pred_data": None
            }, timeout=0.5)

            if step % 5 == 0:
                print(f"  -> 步骤 {step}/{max_steps} 执行完毕...")
                
            time.sleep(0.2) 

        except Exception as e:
            print(f"⚠️ 测试过程中断: {e}")
            break

    print("✅ 闭环测试结束，恢复训练！\n")
    base_model.train()

# ============================================================
# logger
# ============================================================
def get_logger(name="train", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("XVLA Training", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, required=True, help="Path or HF repo for pretrained XVLA")
    parser.add_argument("--output_dir", type=str, default="runnings", help="Directory to save checkpoints")

    # Data
    parser.add_argument("--train_metas_path", type=str, required=True, help="Path to training metadata")
    parser.add_argument("--batch_size", type=int, default=16)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0, help="LR multiplier for soft prompts")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=1000000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)

    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def build_optimizer(model: XVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_soft=1.0):
    """Split param groups by module type with different learning rates."""
    vlm_params = list(model.vlm.parameters())
    soft_prompt_params = list(model.transformer.soft_prompt_hub.parameters())
    action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    exclude = set(map(id, vlm_params + soft_prompt_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "soft_prompts", "params": soft_prompt_params, "lr": lr * lr_coef_soft, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups: 
        if g["name"] == name: g["lr"] = lr


def get_group_lr(optim: torch.optim.Optimizer, name: str) -> float:
    for g in optim.param_groups:
        if g["name"] == name: return g["lr"]
    return 0.0


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Linear warmup followed by cosine decay."""
    if step < start: return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optim, step, args):
    """Elegant group-wise LR scheduler."""
    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
    }
    def schedule(step, base_lr):
        return linear_warmup_cosine(step, args.freeze_steps, args.warmup_steps, args.iters, base_lr, args.min_lr_ratio)
    if step < args.freeze_steps:
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "soft_prompts", base["soft_prompts"])
        set_group_lr(optim, "action_heads", base["action_heads"])
    else:
        for name, base_lr in base.items():
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)
    accelerator = Accelerator(
        log_with="tensorboard", 
        project_dir=output_dir
    )
    accelerator.init_trackers("XVLA-Training")
    
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)
    
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")

    # Load model & processor
    model = XVLA.from_pretrained(args.models)
    
    lora_config = LoraConfig(
        lora_alpha=16,
        r=8,
        bias="none",
        target_modules="all-linear",
        modules_to_save=["transformer.soft_prompt_hub", 
                         "transformer.action_encoder", 
                         "transformer.action_decoder"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    
    processor = XVLAProcessor.from_pretrained(args.models)

    # Iterable dataloader (don't wrap with prepare)
    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
    )

    # Optimizer
    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_soft=args.learning_coef,
    )
    model, optim = accelerator.prepare(model, optim)


    # Training loop
    model.train()
    global_step, t0 = 0, time.time()
    logger.info(f"🚀 Start training for {args.iters} iterations | world_size={accelerator.num_processes}")
    
    
    
    for batch in train_dataloader:
        # Encode language
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
        # Update LR per group
        update_group_lrs(optim, global_step, args)

        # Forward & backward
        # loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss_dict, pred_action, action = model(**inputs)


        # 每 100 步打印一次预测值、真实值和详细的 loss 组成
        if global_step > 0 and global_step % 20 == 0:
            print("pre:++++++++++", pred_action.detach().cpu().numpy())
            print("action:++++++++++", action.detach().cpu().numpy())
            
            # 提取具体的 loss 值，并转化为干净的 Python 浮点数保留 4 位小数
            p_loss = loss_dict.get("position_loss", 0).item()
            r_loss = loss_dict.get("rotate6D_loss", 0).item()
            g_loss = loss_dict.get("gripper_loss", 0).item()
            total_loss = loss.item()
            
            print(f"[Step {global_step}] 📊 Loss Detail => Pos: {p_loss:.4f} | Rot6D: {r_loss:.4f} | Gripper: {g_loss:.4f} | Total: {total_loss:.4f}")
            print("-" * 60) # 加个分割线，方便在终端里肉眼区分


        loss = sum(loss_dict.values())
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()


        if global_step > 0 and global_step % 100 == 0:
            run_closed_loop_rollout(
                model=model, 
                processor=processor, 
                batch_inputs=inputs,  # 直接把 inputs 喂进去就行
                max_steps=30
            )


        # Logging
        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss_sum={logs['loss_total']:.4f} "
                    f"lr_core={logs['lr_transformer_core']:.2e} "
                    f"lr_vlm={logs['lr_vlm']:.2e} ({dt:.2f}s/it)"
                )
        
        # Checkpointing
        global_step += 1
        if accelerator.is_main_process:
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"💾 Saving model to {save_dir}")
                accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump({"global_step": global_step}, f)
        
        if global_step >= args.iters: break

    accelerator.end_training()

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("XVLA training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
