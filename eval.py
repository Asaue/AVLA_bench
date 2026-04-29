import torch
import requests
import time
import base64
import io
import traceback
from PIL import Image
import numpy as np

# 导入你项目专用的 X-VLA 模块
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor

ENV_URL = "http://127.0.0.1:9380"
DOMAIN_ID = 0
INSTRUCTION = "pick up the block and place it on the plate"

# 别忘了在文件最上面加这行导入！
from peft import PeftModel

# ==========================================
# 重新定义路径：基础模型 + 你的 LoRA 权重
# ==========================================
BASE_MODEL_PATH = "/Data/Docker_liuwu/models/xvla-pt"
LORA_PATH = "/Data/Docker_liuwu/models/xvla_check_rec/ckpt-5000"  # 注意你的文件夹名是 ckpt-30000，不是 checkpoint-30000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_processor():
    print(f"⏳ 1. 正在加载基础模型与 Processor: {BASE_MODEL_PATH} ...")
    # 从基础模型路径加载图像处理器和文本分词器
    processor = XVLAProcessor.from_pretrained(BASE_MODEL_PATH)
    
    # 将庞大的基础骨架加载进显存
    base_model = XVLA.from_pretrained(BASE_MODEL_PATH).to(device)
    
    print(f"🔗 2. 正在挂载 LoRA 微调权重: {LORA_PATH} ...")
    # 使用 PEFT 将你训练的 30000 步经验“贴”到基础骨架上
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    model.eval()
    print("✅ 完全体模型与 Processor 加载完毕！")
    return model, processor

def run_standalone_inference():
    model, processor = load_model_and_processor()
    print("🤖 独立 VLA 闭环推理正式启动...")
    
    step = 0
    while True:
        try:
            # --- A. 获取环境真实状态 ---
            resp = requests.get(f"{ENV_URL}/obs", timeout=2.0).json()
            
            # 解析双摄图像
            img_global_bytes = base64.b64decode(resp["image_global_b64"])
            img_global = Image.open(io.BytesIO(img_global_bytes)).convert("RGB")
            
            img_gripper_bytes = base64.b64decode(resp["image_gripper_b64"])
            img_gripper = Image.open(io.BytesIO(img_gripper_bytes)).convert("RGB")
            
            true_ee_pos = resp["ee_pos"]
            true_ee_rot6d = resp.get("ee_rot6d", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            
            # --- B. 组装输入数据 (严格对齐你昨天跑通的逻辑) ---
            # 1. 调用 processor 处理视觉和文本
            # 根据报错记录，你的 processor 接收 (images, text) 格式
            inputs = processor([img_global, img_gripper], INSTRUCTION)
            
            # 2. 组装 Proprio (20维) 和 Domain
            mock_10d_state = list(true_ee_pos) + list(true_ee_rot6d) + [0.0]
            proprio_20d = mock_10d_state + [0.0] * 10
            
            # 3. 将所有输入转为 Tensor 并放到 GPU
            input_ids = inputs["input_ids"].to(device)
            image_input = inputs["image_input"].to(device)
            image_mask = inputs["image_mask"].to(device)
            proprio_tensor = torch.tensor([proprio_20d], dtype=torch.float32).to(device)
            domain_tensor = torch.tensor([[DOMAIN_ID]], dtype=torch.long).to(device)

            # --- C. 模型真实推理 ---
            print(f"[{step}] 🧠 正在思考...")
            with torch.no_grad():
                pred_actions = model.generate_actions(
                    input_ids=input_ids,
                    image_input=image_input,
                    image_mask=image_mask,
                    domain_id=domain_tensor,
                    proprio=proprio_tensor,
                    steps=10
                )
            
            # 拿到真实的预测动作 (取 batch 和 seq 的第一个)
            raw_20d_action = pred_actions[0, 0].cpu().numpy()

            # --- D. 提取 10D Delta 并发送 ---
            raw_10d_delta = raw_20d_action[:10]
            
            # 夹爪指令（>0.5为闭合）
            target_gripper = 1.0 if raw_10d_delta[9] > 0.5 else 0.0
            # 🌟 核心修复：使用 .tolist() 把里面的数字彻底洗成 Python 原生 float
            final_action = raw_10d_delta[:9].tolist() + [float(target_gripper)]

            print(f"📍 真实位置: {np.round(true_ee_pos, 3)} | 🚀 发送增量: {np.round(final_action[:3], 3)}")

            # --- E. 驱动物理引擎 ---
            requests.post(f"{ENV_URL}/action", json={
                "action_type": "ee_pos", 
                "data": final_action
            }, timeout=1.0)

            step += 1
            time.sleep(0.05) 

        except requests.exceptions.Timeout:
            print("⚠️ 连接 MuJoCo 超时！请检查 Windows 服务端是否被鼠标卡住了！")
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ 循环中断:")
            traceback.print_exc() # 打印完整的报错栈，如果模型这里报错能一眼看清
            time.sleep(2)

if __name__ == "__main__":
    run_standalone_inference()