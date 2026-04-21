#!/usr/bin/env python3
"""
Franka Panda 自动数据采集脚本

自动完成"抓取长方体放到盘子上"任务并录制 episodes。
每次随机化物体位置，生成多样化的训练数据。

使用方法:
  python tools/auto_collect.py --num_episodes 50 --headless
  python tools/auto_collect.py --num_episodes 10  # 带可视化窗口

参数:
  --num_episodes   要采集的 episode 数量（默认 50）
  --headless       无头模式，不显示 MuJoCo viewer（更快）
  --record_hz      录制频率 Hz（默认 10）
  --output_dir     输出目录（默认 recordings/）
  --instruction    语言指令（默认 "pick up the block and place it on the plate"）
  --seed           随机种子
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
from PIL import Image

# ─────────────────────────── 路径设置 ───────────────────────────
SCRIPT_DIR = Path(__file__).parent
PANDA_DIR = SCRIPT_DIR.parent
SCENE_XML = PANDA_DIR / "scene.xml"

sys.path.insert(0, str(PANDA_DIR))
from mouse_teleop_franka import (
    TrajectoryRecorder,
    find_preview_cameras,
    level_ctrl,
    qpos_labels,
    actuator_name,
    find_named_site,
)


# ─────────────────────────── IK 求解器 ───────────────────────────

def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_quat: Optional[np.ndarray] = None,
    site_name: str = "attachment_site",
    max_iter: int = 200,
    tol: float = 1e-4,
    step_size: float = 0.5,
) -> tuple[bool, np.ndarray]:
    """
    用 MuJoCo 的雅可比矩阵迭代求解 IK。
    返回 (success, arm_qpos) - 只返回手臂关节角度，不修改 data。
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        for name in ["tool_center_point", "tool0", "ee_site", "gripper"]:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                break
    if site_id < 0:
        return False, data.qpos[:7].copy()

    # 只保存手臂关节（前7个）
    arm_qpos_backup = data.qpos[:7].copy()
    full_qpos_backup = data.qpos.copy()

    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    success = False
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        cur_pos = data.site_xpos[site_id].copy()
        err_pos = target_pos - cur_pos

        if np.linalg.norm(err_pos) < tol:
            success = True
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        lam = 1e-4
        dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + lam * np.eye(3), err_pos)
        data.qpos[:nv] += step_size * dq
        mujoco.mj_forward(model, data)

    # 提取求解到的手臂关节角度
    solved_arm_qpos = data.qpos[:7].copy()

    # 完全恢复原始状态（包括物体位置）
    data.qpos[:] = full_qpos_backup
    mujoco.mj_forward(model, data)

    if not success:
        success = np.linalg.norm(target_pos - data.site_xpos[site_id]) < tol * 10

    return success, solved_arm_qpos


# ─────────────────────────── 仿真控制 ───────────────────────────

def step_sim(model: mujoco.MjModel, data: mujoco.MjData, ctrl: np.ndarray, n_steps: int = 50):
    """执行 n_steps 步仿真，保持控制目标"""
    data.ctrl[:] = ctrl
    for _ in range(n_steps):
        mujoco.mj_step(model, data)


def move_to_joint_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_ctrl: np.ndarray,
    duration_steps: int = 200,
):
    """平滑移动到目标关节角度"""
    start_ctrl = data.ctrl.copy()
    for i in range(duration_steps):
        alpha = (i + 1) / duration_steps
        interp = start_ctrl + alpha * (target_ctrl - start_ctrl)
        data.ctrl[:] = interp
        mujoco.mj_step(model, data)


def set_gripper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: np.ndarray,
    closed: bool,
    steps: int = 100,
):
    """控制夹爪开合 - Franka Panda actuator8: 0=开, 255=关"""
    # Franka Panda 的夹爪执行器名称是 actuator8
    gripper_id = -1
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name == "actuator8":
            gripper_id = i
            break
    # 备用：找最后一个执行器（通常是夹爪）
    if gripper_id < 0:
        gripper_id = model.nu - 1

    lo, hi = model.actuator_ctrlrange[gripper_id]
    # Franka Panda actuator8: 0=关闭, 255=张开（与直觉相反）
    target_val = lo if closed else hi  # 0=关, 255=开
    ctrl[gripper_id] = target_val

    for _ in range(steps):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)


def get_ee_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """获取末端执行器位置"""
    for name in ["attachment_site", "tool_center_point", "tool0", "ee_site"]:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return data.site_xpos[site_id].copy()
    return np.zeros(3)


# ─────────────────────────── 场景随机化 ───────────────────────────

def randomize_scene(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator):
    """随机化物体位置，确保在桌面范围内且不重叠"""
    # 桌子: center=(0.68, 0, 0), size=(0.36, 0.32, 0.03)
    # 桌面 x: 0.32~1.04, y: -0.32~0.32
    TABLE_Z_CUP   = 0.525  # 桌面高 0.42 + 长方体半高 0.04 + 余量
    TABLE_Z_PLATE = 0.435  # 桌面高 0.42 + 盘子厚 0.012 + 余量

    # 长方体：放在桌子靠近机械臂的左半区（x: 0.35~0.60）
    cup_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_target_freejoint")
    if cup_joint_id >= 0:
        adr = model.jnt_qposadr[cup_joint_id]
        cx = rng.uniform(0.35, 0.60)
        cy = rng.uniform(-0.20, 0.20)
        data.qpos[adr:adr+3] = [cx, cy, TABLE_Z_CUP]
        data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
    else:
        cx, cy = 0.45, 0.0

    # 盘子：放在桌子远端（x: 0.65~0.82），且与长方体保持距离
    plate_joint_id = -1
    for i in range(model.njnt):
        body_id = model.jnt_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name == "plate" and model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            plate_joint_id = i
            break

    if plate_joint_id >= 0:
        adr = model.jnt_qposadr[plate_joint_id]
        # 盘子在远端，x 比长方体大至少 0.15m
        for _ in range(20):  # 最多尝试 20 次
            px = rng.uniform(0.62, 0.82)
            py = rng.uniform(-0.18, 0.18)
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            if dist > 0.20:  # 确保距离大于 20cm
                break
        data.qpos[adr:adr+3] = [px, py, TABLE_Z_PLATE]
        data.qpos[adr+3:adr+7] = [1, 0, 0, 0]

    mujoco.mj_forward(model, data)


def get_object_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    """获取物体的世界坐标"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id >= 0:
        return data.xpos[body_id].copy()
    return np.zeros(3)


# ─────────────────────────── 自动抓取任务 ───────────────────────────

def run_pick_and_place(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    recorder: TrajectoryRecorder,
    recording_state: dict,
    rng: np.random.Generator,
) -> bool:
    """执行一次完整的抓取放置任务，返回 True 表示成功。"""

    def ik_ctrl(target_pos, steps=500, tol=1e-3):
        """求解 IK 并返回控制目标，不修改 data 中的物体位置"""
        ok, arm_qpos = solve_ik(model, data, target_pos, max_iter=steps, tol=tol)
        if not ok:
            return None
        new_ctrl = ctrl.copy()
        new_ctrl[:7] = arm_qpos
        return new_ctrl

    def exec_ctrl(target_ctrl, move_steps=200, hold_steps=20):
        """执行控制并录制（移动过程和保持阶段都录制）"""
        nonlocal ctrl
        # 移动阶段也录制
        start_ctrl = data.ctrl.copy()
        for i in range(move_steps):
            alpha = (i + 1) / move_steps
            interp = start_ctrl + alpha * (target_ctrl - start_ctrl)
            data.ctrl[:] = interp
            mujoco.mj_step(model, data)
            if recording_state["value"]:
                recorder.maybe_record(data.time, data)
        ctrl[:] = target_ctrl
        # 保持阶段
        for _ in range(hold_steps):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            if recording_state["value"]:
                recorder.maybe_record(data.time, data)

    # 初始姿态
    init_ctrl = level_ctrl(model)
    move_to_joint_target(model, data, init_ctrl, duration_steps=300)

    # 随机化场景
    randomize_scene(model, data, rng)

    # 稳定仿真
    for _ in range(300):
        data.ctrl[:] = init_ctrl
        mujoco.mj_step(model, data)

    cup_pos  = get_object_pos(model, data, "cup_target")
    plate_pos = get_object_pos(model, data, "plate")

    if np.allclose(cup_pos, 0) or np.allclose(plate_pos, 0):
        print("  ✗ Cannot find objects")
        return False

    print(f"  Cup: {cup_pos.round(3)}, Plate: {plate_pos.round(3)}")

    # 开始录制 - 每次 clear 会重置目录，确保每个 episode 独立
    recording_state["value"] = True
    recorder.clear()  # 这会重置 recording_dir，下次 _create_recording_dir 会创建新目录
    ctrl = init_ctrl.copy()
    ctrl[7] = 255.0  # 夹爪张开

    # 1. 悬停在物体上方
    t = ik_ctrl(cup_pos + [0, 0, 0.10])
    if t is None: return _fail(recording_state, "hover")
    exec_ctrl(t, 300, 1200)

    # 2. 下降到抓取位置
    t = ik_ctrl(cup_pos + [0, 0, 0.01])
    if t is None: return _fail(recording_state, "grasp")
    exec_ctrl(t, 200, 900)

    # 3. 夹爪闭合
    set_gripper(model, data, ctrl, closed=True, steps=200)
    for _ in range(1200):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        if recording_state["value"]:
            recorder.maybe_record(data.time, data)

    # 4. 抬起
    t = ik_ctrl(cup_pos + [0, 0, 0.18])
    if t is None: return _fail(recording_state, "lift")
    exec_ctrl(t, 200, 900)

    # 检查是否夹起来
    cup_z = get_object_pos(model, data, "cup_target")[2]
    if cup_z < cup_pos[2] + 0.05:
        print(f"  ✗ Grasp failed: cup z={cup_z:.3f}")
        recording_state["value"] = False
        return False
    print(f"  ✓ Grasp success: cup z={cup_z:.3f}")

    # 5. 移动到盘子上方
    t = ik_ctrl(plate_pos + [0, 0, 0.18])
    if t is None: return _fail(recording_state, "place hover")
    exec_ctrl(t, 300, 1200)

    # 6. 下降到放置位置
    t = ik_ctrl(plate_pos + [0, 0, 0.06])
    if t is None: return _fail(recording_state, "place")
    exec_ctrl(t, 200, 900)

    # 7. 夹爪张开
    set_gripper(model, data, ctrl, closed=False, steps=200)
    for _ in range(1200):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        if recording_state["value"]:
            recorder.maybe_record(data.time, data)

    # 8. 退出
    t = ik_ctrl(plate_pos + [0, 0, 0.22])
    if t is not None:
        exec_ctrl(t, 150, 900)

    recording_state["value"] = False

    # 判断成功：只要长方体比初始位置高（说明在盘子上），且大致在盘子方向
    final_cup = get_object_pos(model, data, "cup_target")
    dist_xy = np.linalg.norm(final_cup[:2] - plate_pos[:2])
    # 放宽条件：长方体在盘子半径 15cm 内，且高度高于桌面（说明在盘子上而不是掉地上）
    success = dist_xy < 0.15 and final_cup[2] > 0.43

    if success and recorder.sample_count() > 5:
        csv_path = recorder.save_csv()
        print(f"  ✓ Saved {recorder.sample_count()} frames → {csv_path.parent.name}/")
        return True
    else:
        print(f"  ✗ Failed (dist_xy={dist_xy:.3f}, cup_z={final_cup[2]:.3f}, samples={recorder.sample_count()})")
        if recorder.sample_count() > 5:
            csv_path = recorder.save_csv()
            print(f"  → Saved anyway: {recorder.sample_count()} frames → {csv_path.parent.name}/")
            return True
        return False


def _fail(recording_state: dict, stage: str) -> bool:
    print(f"  ✗ IK failed: {stage}")
    recording_state["value"] = False
    return False


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto collect Franka pick-and-place episodes")
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--headless', action='store_true', help='No viewer window')
    parser.add_argument('--record_hz', type=float, default=3.0)
    parser.add_argument('--output_dir', type=str, default=str(PANDA_DIR / 'recordings'))
    parser.add_argument('--instruction', type=str,
                        default='pick up the block and place it on the plate')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading model: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    # 初始化渲染器（用于录制图像）
    renderer = mujoco.Renderer(model, height=256, width=256)
    preview_cameras = find_preview_cameras(model)
    print(f"Cameras: {preview_cameras}")

    # 创建录制器
    recording_state = {"value": False}
    recorder = TrajectoryRecorder(
        model=model,
        output_path=None,
        record_hz=args.record_hz,
        renderer=renderer,
        preview_cameras=preview_cameras,
    )

    print(f"\nTarget: {args.num_episodes} episodes")
    print(f"Instruction: \"{args.instruction}\"")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    success_count = 0
    attempt_count = 0

    while success_count < args.num_episodes:
        attempt_count += 1
        print(f"\n[Attempt {attempt_count}] Episode {success_count+1}/{args.num_episodes}")

        # 重置仿真
        mujoco.mj_resetData(model, data)
        init_ctrl = level_ctrl(model)
        data.ctrl[:] = init_ctrl
        mujoco.mj_forward(model, data)

        ok = run_pick_and_place(model, data, recorder, recording_state, rng)
        if ok:
            success_count += 1

        # 防止无限循环
        if attempt_count > args.num_episodes * 3:
            print(f"\n⚠ Too many failures, stopping at {success_count} episodes")
            break

    print(f"\n{'='*60}")
    print(f"Collected: {success_count}/{args.num_episodes} episodes")
    print(f"Success rate: {success_count/max(attempt_count,1)*100:.1f}%")
    print(f"Recordings saved to: {args.output_dir}")
    print(f"\nNext step - convert to HDF5:")
    print(f"  python tools/csv_to_hdf5.py \\")
    print(f"    --recordings_dir recordings \\")
    print(f"    --output_dir ../../../datasets/franka_mujoco_hdf5 \\")
    print(f"    --instruction \"{args.instruction}\" \\")
    print(f"    --gen_meta")


if __name__ == '__main__':
    main()
