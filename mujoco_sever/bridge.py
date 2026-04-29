import requests
import time
import base64
import numpy as np
import cv2
import json_numpy

ENV_URL = "http://127.0.0.1:9380"
VLA_URL = "http://127.0.0.1:8010"

INSTRUCTION = "pick up the block and place it on the plate" 
DOMAIN_ID = 0

def run_loop():
    print("🤖 20维 VLA 闭环推理启动 (完美适配增量版)...")
    
    while True:
        try:
            # 1. 获取观察
            obs = requests.get(f"{ENV_URL}/obs").json()
            
            img_grip_bytes = base64.b64decode(obs["image_gripper_b64"])
            img_grip_arr = cv2.imdecode(np.frombuffer(img_grip_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            img_glob_bytes = base64.b64decode(obs["image_global_b64"])
            img_glob_arr = cv2.imdecode(np.frombuffer(img_glob_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            true_ee_pos = np.array(obs["ee_pos"])
            
            # [安全兜底] 如果服务端 /obs 还没传真正的 6D 姿态，先用默认的顶替，凑齐 10D
            true_ee_rot6d = obs.get("ee_rot6d", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            gripper_val = [0.0] 
            
            # 拼接 20D 状态
            mock_10d_state = list(true_ee_pos) + list(true_ee_rot6d) + gripper_val
            proprio_20d = mock_10d_state + [0.0] * 10
            
            payload = {
                "image0": json_numpy.dumps(img_glob_arr),  # 0是全局
                "image1": json_numpy.dumps(img_grip_arr),  # 1是夹爪
                "language_instruction": INSTRUCTION,
                "proprio": json_numpy.dumps(np.array(proprio_20d)),
                "domain_id": DOMAIN_ID,
                "steps": 10
            }

            # 2. 大脑推理
            vla_resp = requests.post(f"{VLA_URL}/act", json=payload).json()
            if "error" in vla_resp:
                continue

            action_20d = vla_resp["action"]
            if isinstance(action_20d[0], list): action_20d = action_20d[0]
                
            # 3. 提取 Delta 动作
            # 💡 核心 1：直接使用模型输出的原始增量（不进行多余的反归一化）
            raw_10d_delta = action_20d[:10]
            
            # 💡 核心 2：直接把包含 6D 姿态在内的 10维 Delta 组装好
            # 夹爪指令（通常模型输出 > 0.5 表示闭合）
            target_gripper = 1.0 if raw_10d_delta[9] > 0.5 else 0.0
            
            # 前 9 维是位置+姿态增量，第 10 维是绝对的夹爪指令
            final_action = list(raw_10d_delta[:9]) + [target_gripper]

            print(f"📍 真实位置: {np.round(true_ee_pos, 3)} | 🧠 发送 Delta: {np.round(final_action[:3], 3)}")

            # 💡 核心 3：把纯 Delta 传给服务端（因为服务端内部已经写了 Target = Current + Delta）
            requests.post(f"{ENV_URL}/action", json={
                "action_type": "ee_pos", 
                "data": final_action
            })

            time.sleep(0.05)

        except Exception as e:
            print(f"⚠️ 循环中断或连接失败: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run_loop()