import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import base64
import cv2
import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# ─────────────────────────── 路径与服务配置 ───────────────────────────
SCRIPT_DIR = Path(__file__).parent
SCENE_XML = SCRIPT_DIR / "config" / "scene.xml"
HOME_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
app = FastAPI()

class SimState:
    def __init__(self):
        self.target_ctrl = None
        self.latest_img = ""
        self.current_qpos = []
        self.pending_action = None 
        self.lock = threading.Lock()
        # 👇 必须加这个：告诉主线程需要重置的信号旗帜
        self.needs_reset = False 

state = SimState()

# 【核心修改】：优化数据结构，完美兼顾训练和推理
class Action(BaseModel):
    action_type: str
    data: Optional[List[float]] = None       # 控制实体手臂的数据 (训练时传GT，推理时传Pred)
    pred_data: Optional[List[float]] = None  # 控制幽灵指示器的数据 (仅用于显示，如果传了就更新幽灵位置)

# ─────────────────────────── 核心 IK 算法 ───────────────────────────
def rot6d_to_mat(rot6d):
    """将 6D 旋转向量转换为 3x3 旋转矩阵"""
    rot6d = np.array(rot6d)
    x = rot6d[0:3]
    y = rot6d[3:6]
    # 正交化(Gram-Schmidt)
    x_norm = np.linalg.norm(x)
    if x_norm > 1e-6: x = x / x_norm
    y = y - np.dot(x, y) * x
    y_norm = np.linalg.norm(y)
    if y_norm > 1e-6: y = y / y_norm
    z = np.cross(x, y)
    return np.column_stack((x, y, z))

def solve_ik(model, data, target_pos, target_rot6d=None, site_name="attachment_site", max_iter=200, tol=1e-4, step_size=0.5):
    """支持 3 自由度(仅位置)或 6 自由度(位置+姿态)的 IK 求解器"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        for name in ["tool_center_point", "tool0", "ee_site", "gripper"]:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0: break
    if site_id < 0:
        return False, data.qpos[:7].copy()

    full_qpos_backup = data.qpos.copy()
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    # 解析目标旋转矩阵
    R_target = rot6d_to_mat(target_rot6d) if target_rot6d is not None else None

    success = False
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        cur_pos = data.site_xpos[site_id].copy()
        err_pos = target_pos - cur_pos

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        if R_target is not None:
            # === 6 自由度 IK (位置 + 姿态) ===
            R_cur = data.site_xmat[site_id].reshape(3, 3)
            # 计算姿态误差 (通过列向量的叉乘近似角速度误差)
            err_rot = 0.5 * (np.cross(R_cur[:, 0], R_target[:, 0]) +
                             np.cross(R_cur[:, 1], R_target[:, 1]) +
                             np.cross(R_cur[:, 2], R_target[:, 2]))
            
            if np.linalg.norm(err_pos) < tol and np.linalg.norm(err_rot) < tol:
                success = True
                break
                
            err_6d = np.hstack([err_pos, err_rot])
            jac_6d = np.vstack([jacp, jacr])
            
            lam = 1e-4
            dq = jac_6d.T @ np.linalg.solve(jac_6d @ jac_6d.T + lam * np.eye(6), err_6d)
        else:
            # === 3 自由度 IK (仅位置) ===
            if np.linalg.norm(err_pos) < tol:
                success = True
                break
                
            lam = 1e-4
            dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + lam * np.eye(3), err_pos)

        data.qpos[:nv] += step_size * dq

    solved_arm_qpos = data.qpos[:7].copy()
    data.qpos[:] = full_qpos_backup  
    mujoco.mj_forward(model, data)
    return success, solved_arm_qpos

# ─────────────────────────── HTTP 接口定义 ───────────────────────────
@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.get("/reset")
def reset_env():
    """安全重置环境：API 绝对不碰 MuJoCo，只发信号"""
    with state.lock:
        state.needs_reset = True  # 升起重置旗帜
        state.pending_action = None
    
    # 最多等待 0.5 秒，看主线程有没有把旗帜降下来（重置完毕）
    for _ in range(50):
        if not state.needs_reset:
            break
        time.sleep(0.01)
        
    return {"status": "success"}

@app.get("/obs")
def get_obs():
    with state.lock:
        return {
            "image_gripper_b64": getattr(state, 'img_gripper_b64', ""),
            "image_global_b64": getattr(state, 'img_global_b64', ""),
            "qpos": state.current_qpos,
            "ee_pos": getattr(state, 'current_ee_pos', [0.45, 0.0, 0.5]) 
        }

@app.post("/action")
def apply_action(action: Action):
    """API 线程：只负责把指令放进邮箱，绝不动底层物理"""
    with state.lock:
        state.pending_action = action
        print("++++++++action:",action)
    return {"status": "success"}

# ─────────────────────────── MuJoCo 仿真主循环 ───────────────────────────
def run_sim():
    global model, data
    if not SCENE_XML.exists():
        print(f"❌ 找不到模型文件，请检查路径: {SCENE_XML}")
        return

    print(f"✅ 成功加载模型: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=224, width=224)
    
    # ==========================================
    # 🌟 核心修改：程序刚启动时的“第一帧”绝对初始化
    # ==========================================
    mujoco.mj_resetData(model, data)
    
    # 强行覆盖初始姿态！
    data.qpos[:7] = HOME_QPOS  
    # 如果你的夹爪默认是闭合的(贴在一起)，可以在这里加一行让夹爪默认张开
    data.qpos[7:9] = [0.04, 0.04]  # 根据你夹爪的实际控制数值调整
    
    mujoco.mj_forward(model, data) # 刷新物理状态
    
    # 初始化控制器，防止一开机机械臂被拽回原点
    init_ctrl = np.zeros(model.nu)
    init_ctrl[:7] = data.qpos[:7]
    init_ctrl[7] = 255
    data.ctrl[:] = init_ctrl
    state.target_ctrl = init_ctrl.copy()

    print("🚀 正在启动 3D 交互窗口...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_render = time.time()
        ghost_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ghost_ee")
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        if ghost_b_id >= 0 and site_id >= 0:
            mocap_id = model.body_mocapid[ghost_b_id]
            data.mocap_pos[mocap_id] = data.site_xpos[site_id].copy()
            mujoco.mj_forward(model, data) # 刷新一下物理状态
        while viewer.is_running():
            step_start = time.time()
            
            # ==========================================
            # 🌟 1. 物理线程自己接管重置，绝对安全不报错
            # ==========================================
            with state.lock:
                if state.needs_reset:
                    mujoco.mj_resetData(model, data)
                    
                    # 👇 新增：强制设置初始关节姿态！
                    data.qpos[:7] = HOME_QPOS
                    
                    # 必须立刻 forward 一次，让物理引擎计算新的空间坐标和相机视角
                    mujoco.mj_forward(model, data)
                    
                    init_ctrl = np.zeros(model.nu)
                    init_ctrl[:7] = data.qpos[:7]  
                    init_ctrl[7] = 255
                    data.ctrl[:] = init_ctrl
                    state.target_ctrl = init_ctrl.copy()
                    
                    state.needs_reset = False # 降下旗帜
                    continue # 跳过当前帧，重新开始

            # 🌟 2. 检查邮箱是否有新指令
            pending = None
            with state.lock:
                if state.pending_action is not None:
                    pending = state.pending_action
                    state.pending_action = None
            
            # 2. 物理动作解算
            if pending:
                # --- A. 实体手臂控制 (驱动物理引擎) ---
                if pending.data is not None:
                    action_arr = np.array(pending.data)
                    if pending.action_type == "joint":
                        state.target_ctrl[:7] = action_arr[:7]
                        state.target_ctrl[7] = action_arr[7] 
                    elif pending.action_type == "ee_pos":
                        # 1. 提取所有的增量数据
                        delta_xyz = action_arr[0:3]        # 位置增量
                        delta_rot6d = action_arr[3:9]      # 🌟 姿态 6D 增量
                        gripper_val = action_arr[9]        # 夹爪指令
                        # gripper_val = 1
                        # 2. 获取当前末端的【真实绝对坐标】和【真实绝对 6D 姿态】
                        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
                        if site_id >= 0:
                            current_xyz = data.site_xpos[site_id].copy()
                            # 读取当前的 3x3 旋转矩阵
                            current_rmat = data.site_xmat[site_id].reshape(3, 3)
                            # 按照你的生成逻辑，提取前两列作为当前的 6D 连续表示
                            current_rot6d = np.concatenate([current_rmat[:, 0], current_rmat[:, 1]])
                        else:
                            # 兜底默认值
                            current_xyz = np.array([0.45, 0.0, 0.5])
                            current_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) # 单位矩阵的 6D 表示
                        
                        # 🌟 3. 核心数学：目标 = 当前 + 增量
                        target_xyz = current_xyz + delta_xyz
                        target_rot6d = current_rot6d + delta_rot6d
                        
                        # 4. 解开封印！把 XYZ 和 姿态 同时喂给 IK 求解器
                        success, arm_qpos = solve_ik(
                            model, data, 
                            target_pos=target_xyz, 
                            target_rot6d=target_rot6d  # 👈 这里传的是计算后的绝对目标姿态！
                        ) 
                        
                        if success:
                            state.target_ctrl[:7] = arm_qpos
                        else:
                            print(f"⚠️ IK 求解失败！目标偏离过大 -> Pos: {target_xyz}")
                        
                        # 5. 夹爪控制逻辑 
                        mapped_ctrl = np.clip((gripper_val / 0.04) * 255.0, 0, 255)
                        state.target_ctrl[7] = mapped_ctrl

                # --- B. 幽灵指示器控制 (无物理碰撞，仅移动 mocap) ---
                if pending.pred_data is not None and pending.action_type == "ee_pos":
                    pred_arr = np.array(pending.pred_data)
                    pred_xyz = pred_arr[:3]
                    pred_rot6d = pred_arr[3:9]
                    
                    # 寻找我们在 scene.xml 里的幽灵实体
                    ghost_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ghost_ee")
                    if ghost_b_id >= 0:
                        mocap_id = model.body_mocapid[ghost_b_id]
                        if mocap_id >= 0:
                            # 1. 瞬移到预测位置
                            data.mocap_pos[mocap_id] = pred_xyz
                            
                            # 2. 瞬移到预测姿态 (转为四元数)
                            R_pred = rot6d_to_mat(pred_rot6d)
                            quat = np.zeros(4)
                            mujoco.mju_mat2Quat(quat, R_pred.flatten())
                            data.mocap_quat[mocap_id] = quat

            # 3. 物理步进
            data.ctrl[:] = state.target_ctrl
            mujoco.mj_step(model, data)
            viewer.sync()

            # 4. 渲染相机存入后台
            if time.time() - last_render > 0.1:
                # --- 拍第一张：夹爪视角 ---
                try:
                    renderer.update_scene(data, camera="cam_ee") 
                except Exception:
                    renderer.update_scene(data)
                img_gripper = renderer.render()
                cv2.imwrite("mujoco_vision_debug1.jpg", cv2.cvtColor(img_gripper, cv2.COLOR_RGB2BGR))
                _, buf_w = cv2.imencode('.jpg', cv2.cvtColor(img_gripper, cv2.COLOR_RGB2BGR))
                
                # --- 拍第二张：全局视角 ---
                try:
                    renderer.update_scene(data, camera="cam_global")
                except Exception:
                    renderer.update_scene(data)
                img_global = renderer.render()
                # 注意：MuJoCo 渲染出来是 RGB，cv2 保存需要 BGR
                cv2.imwrite("mujoco_vision_debug2.jpg", cv2.cvtColor(img_global, cv2.COLOR_RGB2BGR))
                _, buf_g = cv2.imencode('.jpg', cv2.cvtColor(img_global, cv2.COLOR_RGB2BGR))
                
                # 获取真实末端坐标
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
                if site_id >= 0:
                    true_ee_pos = data.site_xpos[site_id].tolist()
                else:
                    true_ee_pos = [0.45, 0.0, 0.5]

                with state.lock:
                    state.img_gripper_b64 = base64.b64encode(buf_w).decode('utf-8')
                    state.img_global_b64 = base64.b64encode(buf_g).decode('utf-8')
                    state.current_qpos = data.qpos[:7].tolist()
                    state.current_ee_pos = true_ee_pos
                last_render = time.time()
            
            # 🌟 3. 循环最底部的速度控制（极其重要）
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            else:
                # 🌟 极其核心的救命代码：
                # 即使渲染导致掉帧，也必须强行 sleep 1 毫秒！
                # 强迫主线程交出 Python GIL，让后台的 FastAPI 线程能处理网络请求！
                time.sleep(0.001)

import asyncio
import uvicorn

def start_api():
    """使用 asyncio 启动，完美绕过 Windows 子线程无法接收信号的报错"""
    config = uvicorn.Config(app, host="0.0.0.0", port=9380, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())

if __name__ == "__main__":
    # 启动后台 API 线程
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    # 启动前台物理仿真
    run_sim()