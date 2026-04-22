# 🤖 AVLA (AVLA Bench)
本仓库是一个通用的视觉-语言-动作 (VLA) 模型训练、评估与部署框架。现已全面支持自定义 Franka 仿真数据 的快速接入与微调。

## 🛠️ 1. 环境配置
```Bash
conda env create -f environment.yml
conda activate xvla_env
```

## 📦 2. 数据准备
请确保你的 Franka 仿真数据已通过 datasets/domain_handler/franka_mujoco_handler.py 处理为标准格式，并生成了对应的 meta.json 索引文件。

## 🚀 3. 模型微调 (Franka Simulation)
使用以下命令，基于你准备好的 Franka 仿真数据集对 XVLA 进行微调：

```Bash
accelerate launch \
    --num_processes=1 \
    --mixed_precision bf16 \
    train.py \
    --models '……models/xvla-pt' \
    --train_metas_path '……X-VLA-main/datasets/franka_xvla_ready/meta.json' \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --iters 500 \
    --freeze_steps 50 \
    --warmup_steps 100 \
    --save_interval 250 \
    --output_dir '……/models/xvla_check'
```

提示：显存不足时，可使用 peft_train.py 替代 train.py 进行 LoRA 微调。

## 🔌 4. 模型部署测试
训练完成后，可以直接启动服务端加载最新的 Checkpoint 进行测试：

```Bash
python deploy.py --model_path ……/models/xvla_check
```