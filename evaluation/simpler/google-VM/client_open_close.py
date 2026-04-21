import argparse
import os
import time
import json
import json_numpy
import requests
import numpy as np
import torch
import math
import collections
from scipy.spatial.transform import Rotation as R
from PIL import Image
from sapien.core import Pose
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video
import simpler_env
import sys
from itertools import product
from transforms3d.euler import euler2quat
import itertools
from pathlib import Path


# ======================================================
# === Utility: Environment Config ====================
# ======================================================

SIMPLER_DIR = os.getenv("SIMPLER_DIR", "/default/path/if/not/set")

CONFIG_PATH = Path(__file__).parent / "configs/open_close.json"

def _apply_env_placeholders(obj):
    if isinstance(obj, dict):
        return {k: _apply_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_apply_env_placeholders(v) for v in obj]
    if isinstance(obj, str):
        return obj.replace("{SIMPLER_DIR}", SIMPLER_DIR)
    return obj

def parse_range_tuple(t):
    if isinstance(t, (int, float)):
        return [t]
    return np.linspace(t[0], t[1], int(t[2])).tolist()

def generate_robot_init_quats(quat_center, rpy_range):
    r_range = parse_range_tuple(rpy_range[:3])
    p_range = parse_range_tuple(rpy_range[3:6])
    y_range = parse_range_tuple(rpy_range[6:])
    return [
        (Pose(q=euler2quat(r, p, y)) * Pose(q=quat_center)).q
        for r, p, y in product(r_range, p_range, y_range)
    ]
    
# ======================================================
# === Utility: Rotation conversions ====================
# ======================================================

def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to 6D rotation representation."""
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def rotate6D_to_euler_xyz(v6: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation back to Euler angles (xyz)."""
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError(f"Last dimension must be 6, got {v6.shape[-1]}")
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    return R.from_matrix(rot_mats).as_euler("xyz")

# ======================================================
# === HTTP Client for XVLA FastAPI server ==============
# ======================================================
class XVLAClient:
    """
    Lightweight HTTP client that queries an XVLA FastAPI server for action predictions.
    """

    def __init__(self, host: str, port: int, timeout: int = 20):
        self.url = f"http://{host}:{port}/act"
        self.timeout = timeout
        self.reset()

    def reset(self, proprio=None, instruction=None, current_xyz=None):
        self.proprio = proprio
        self.instruction = instruction
        self.action_plan = collections.deque()
        self.current_xyz = current_xyz

    def set_instruction(self, instruction: str):
        self.instruction = instruction

    def step(self, image: np.ndarray) -> np.ndarray:
        """
        Query the XVLA model server for next action given the current image.

        Returns:
            np.ndarray of shape (D_action,)
        """
        if not self.action_plan:
            payload = {
                "proprio": json_numpy.dumps(self.proprio),
                "language_instruction": self.instruction,
                "image0": json_numpy.dumps(image),
                "domain_id": 1,
                "steps": 10,
            }
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                action_seq = np.array(result["action"], dtype=np.float32)[::2][:10] # action speedup and chunk cutting
                action_seq[:, :3] += self.current_xyz
                self.action_plan.extend(action_seq.tolist())
            except Exception as e:
                print(f"[Client] Request failed: {e}")
                return np.zeros_like(self.proprio)

        action_pred = np.array(self.action_plan.popleft(), dtype=np.float32)

        # Postprocess 6D rotation -> Euler xyz + gripper binary
        action_final = np.concatenate([
            action_pred[:3],
            rotate6D_to_euler_xyz(action_pred[3:9]),
            np.array([1.0 if action_pred[9] > 0.35 else -1.0])
        ])
        self.current_xyz = action_final[:3]
        return action_final

# ======================================================
# === Google evaluation routine ========================
# ======================================================
def evaluate_policy_Google(client, output_dir: str, current_scenario_config: dict, scenario_name: str, max_steps: int = 1200):

    max_steps = current_scenario_config["max_episode_steps"] * 2 # Here we use 2x environment steps
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "google_results.txt")
    summary_path = os.path.join(output_dir, "google_summary.txt")
    summary = []
    
    print("\n" + "=" * 70)
    print(f"🧩 [Eval] Scenario: {scenario_name} | Max Steps: {max_steps}")
    print("=" * 70)
    
    current_scenario_config["control_freq"] = 3
    current_scenario_config["sim_freq"] = 513
    robot_init_quats = generate_robot_init_quats(
        current_scenario_config["robot_init_rot_quat_center"], 
        current_scenario_config["robot_init_rot_rpy_range"]
    )
    if "rgb_overlay_path" in current_scenario_config:
        if "rgb_overlay_cameras" not in current_scenario_config:
            if "google_robot_static" in current_scenario_config["robot_name"]:
                current_scenario_config["rgb_overlay_cameras"] = ["overhead_camera"]

    ep_count = 0
    success_count = 0
    for robot_init_x in parse_range_tuple(current_scenario_config["robot_init_x"]):
        current_scenario_config["robot_init_x"] = robot_init_x
        for robot_init_y in parse_range_tuple(current_scenario_config["robot_init_y"]):
            current_scenario_config["robot_init_y"] = robot_init_y
            for robot_init_quat in robot_init_quats:
                current_scenario_config["robot_init_rot_quat"] = robot_init_quat
                
                make_kwargs = dict(
                    robot=current_scenario_config["robot_name"],
                    sim_freq=current_scenario_config["sim_freq"],
                    control_freq=current_scenario_config["control_freq"],
                    control_mode="arm_pd_ee_base_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
                    scene_name=current_scenario_config["scene_name"],
                    camera_cfgs={"add_segmentation": True},
                    rgb_overlay_path=current_scenario_config.get("rgb_overlay_path", None),
                    rgb_overlay_cameras=current_scenario_config.get("rgb_overlay_cameras", None),
                )

                images = []
                env = simpler_env.make(current_scenario_config["env_name"], **make_kwargs, **current_scenario_config["additional_env_build_kwargs"])
                options = {
                    "robot_init_options": {
                        "init_xy": np.array([current_scenario_config["robot_init_x"], current_scenario_config["robot_init_y"]]),
                        "init_rot_quat": robot_init_quat,
                    }
                }
                
                reset_combinations = []
                if current_scenario_config["obj_variation_mode"] == "episode":
                    for ep_id in range(current_scenario_config["episode_nums"]):
                        reset_combinations.append(
                            {
                                "episode_id": ep_id
                            })
                elif current_scenario_config["obj_variation_mode"] == "xy":
                    x_list = parse_range_tuple(current_scenario_config["obj_init_x_range"])
                    y_list = parse_range_tuple(current_scenario_config["obj_init_y_range"])
                    xy_combinations = list(itertools.product(x_list, y_list))
                    for x, y in xy_combinations:
                        reset_combinations.append(
                            {
                                "init_xy": np.array([x, y])
                            })
                        
                # Above is preparation for Simpler environment options
                # main loop
                for obj_reset_option in reset_combinations:
                    options["obj_init_options"] = obj_reset_option
                    obs, _ = env.reset(options=options)
                    print(f"Eval scenario: {scenario_name} for {ep_count}-th episode......")
                    images = []
                    instruction = env.get_language_instruction()
                    print(f"📝 Now Instruction: {instruction}")
                    
                    proprio = torch.zeros(20).to(dtype=torch.float32).numpy()
                    ee_pose_wrt_base = Pose(p=obs['agent']['base_pose'][:3], q=obs['agent']['base_pose'][3:]).inv() * Pose(p=obs['extra']['tcp_pose'][:3], q=obs['extra']['tcp_pose'][3:])
                    current_xyz = torch.tensor(ee_pose_wrt_base.p).cuda().view(1, 3)
                    # Reset XVLA client
                    client.reset(proprio, instruction, current_xyz.cpu().numpy())
                    
                    # === Run environment loop ===
                    task_start = time.time()
                    for step_idx in range(max_steps):
                        instruction = env.get_language_instruction()
                        if instruction != client.instruction:
                            client.set_instruction(instruction)
                            print(f"📝 Now Instruction: {instruction}")
                        image = get_image_from_maniskill2_obs_dict(env, obs)

                        action = client.step(image)
                        obs, reward, done, _, _ = env.step(action)
                        images.append(image.copy())

                        if done:
                            print(f"✅ Scenario {scenario_name} completed in {step_idx+1} steps (suc={done})")
                            break
                        
                    # === Save video & log ===
                    duration = time.time() - task_start
                    out_video = os.path.join(output_dir, f"{scenario_name}_{ep_count}_{done:.2f}.mp4")
                    write_video(out_video, images, fps=10)

                    result = {
                        "scenario": scenario_name,
                        "episode_id": ep_count,
                        "reward": float(reward),
                        "done": bool(done),
                        "steps": step_idx + 1,
                        "duration_sec": duration,
                        "output": out_video,
                    }
                    summary.append(result)
                    if done:
                        success_count += 1

                    with open(log_path, "a+") as f:
                        f.write(json.dumps(result) + "\n")

                    print(f"🎥 Saved video to {out_video}")
                    print(f"🕒 Current episode duration: {duration:.1f}s")
                    ep_count += 1
    
    result = {
        "scenario": scenario_name,
        "total_episodes": ep_count,
        "success_count": success_count,
        "success_rate": success_count / ep_count,
    }
    with open(summary_path, "a+") as f:
        f.write(json.dumps(result) + "\n")

    
def main():
    parser = argparse.ArgumentParser("XVLA Google Robot Evaluation Client")
    parser.add_argument("--connection_info", type=str, default=None,
                        help="Path to server info.json written by XVLA server")
    parser.add_argument("--server_ip", type=str, default=None,
                        help="Manual server IP (if not using connection_info)")
    parser.add_argument("--server_port", type=int, default=None,
                        help="Manual server port (if not using connection_info)")
    parser.add_argument("--output_dir", type=str, default="logs/open_close/",
                        help="Directory for saving evaluation videos and logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        with open(CONFIG_PATH, "r") as f:
            _raw_cfg = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] Environment specific config not found at {CONFIG_PATH}")
        raise ValueError(f"Config file not found: {CONFIG_PATH}")
    env_dict = _apply_env_placeholders(_raw_cfg)

    print("🚀 [Client] Starting XVLA evaluation client...")

    # ------------------------------------------------------------------
    # 1. Load connection info
    # ------------------------------------------------------------------
    if args.connection_info is not None:
        print(f"🔍 Waiting for connection info file: {args.connection_info}")
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not os.path.exists(args.connection_info):
            sys.stdout.write(f"\r{spinner[i % len(spinner)]} Waiting for server to start...")
            sys.stdout.flush()
            time.sleep(0.5)
            i += 1
        print("\n✅ Connection info file found!")
        try:
            with open(args.connection_info, "r") as f:
                infos = json.load(f)
            host, port = infos["host"], infos["port"]
            print(f"🔗 Loaded server info: host={host}, port={port}")
        except Exception as e:
            print(f"❌ Failed to read connection info: {e}")
            sys.exit(1)
    else:
        if not args.server_ip or not args.server_port:
            print("❌ Must specify either --connection_info or both --server_ip and --server_port.")
            sys.exit(1)
        host, port = args.server_ip, args.server_port
        print(f"🔗 Using manual server address: {host}:{port}")

    # ------------------------------------------------------------------
    # 2. Connect to server
    # ------------------------------------------------------------------
    print(f"🛰️  Connecting to XVLA server at {host}:{port} ...")
    client = XVLAClient(host, port)
    print("✅ Successfully initialized XVLA client!")

    # ------------------------------------------------------------------
    # 3. Run evaluation
    # ------------------------------------------------------------------
    print("🎯 Starting Google Robot Evaluation Client...")
    print(f"📁 Results and videos will be saved to: {os.path.abspath(args.output_dir)}")

    scenario_nums = len(env_dict)
    count = 0
    for scenario_name in env_dict.keys():
        current_scenario_config = env_dict[scenario_name]
        print(f"\n--- 🧩 Starting evaluation process {count}/{scenario_nums} ---")
        count += 1
        try:
            evaluate_policy_Google(client, args.output_dir, current_scenario_config, scenario_name)
        except KeyboardInterrupt:
            print("🛑 Interrupted by user. Exiting gracefully...")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️ Process {count} failed with error: {e}")
            continue

    print("\n✅ All evaluations completed successfully!")
    print(f"🎥 Check your videos and logs under: {os.path.abspath(args.output_dir)}")

# ======================================================
# === Entry ============================================
# ======================================================
if __name__ == "__main__":
    main()
