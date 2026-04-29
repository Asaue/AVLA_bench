import requests
import time

# # 如果你在 GPU6 上跑且做了端口转发，或者直接在 Windows 本地跑，URL 都是这个
# URL = "http://127.0.0.1:9380"

# def test_vla_connection():
#     print("📡 1. 测试获取环境观察 (Observation)...")
#     try:
#         obs_resp = requests.get(f"{URL}/obs")
#         obs_resp.raise_for_status()
#         obs_data = obs_resp.json()
#         print(f"✅ 成功！获取到当前关节角度: {obs_data['qpos']}")
#         print(f"✅ 成功！获取到图片 Base64 长度: {len(obs_data['image_b64'])}")
#     except Exception as e:
#         print(f"❌ 获取失败，请检查服务是否运行: {e}")
#         return

#     print("\n🚀 2. 发送测试动作 (Action)...")
#     print("指令：移动末端到杯子正上方 (x=0.45, y=0.0, z=0.6)，并张开夹爪")
    
#     # 构造动作数据，调用你服务器里的 ee_pos (末端坐标) 逆解算模式
#     payload = {
#         "action_type": "ee_pos",
#         "data": [0.45, 0.0, 0.6, 255.0]  # [X, Y, Z, 夹爪开度]
#     }

#     try:
#         act_resp = requests.post(f"{URL}/action", json=payload)
#         print(f"✅ 服务器响应: {act_resp.json()}")
#         print("👀 快看你的 3D 窗口，机械臂应该动起来了！")
#     except Exception as e:
#         print(f"❌ 动作发送失败: {e}")

# if __name__ == "__main__":
#     test_vla_connection()

import requests
import time

print("开始测试连接...")
start_time = time.time()
try:
    # 测试获取观测数据（图片+状态）
    resp = requests.get("http://127.0.0.1:9380/ping", timeout=5.0)
    print(f"✅ 连接成功！耗时: {time.time() - start_time:.2f} 秒")
    print(f"服务端返回内容: {resp.text}")
except requests.exceptions.Timeout:
    print("❌ 连接超时！服务端还活着，但是响应时间超过了设定的阈值。")
except requests.exceptions.ConnectionError:
    print("❌ 连接拒绝！服务端已经崩溃或没有启动。")