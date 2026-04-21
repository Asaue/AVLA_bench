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

    def reset(self, proprio=None, instruction=None):
        self.proprio = proprio
        self.instruction = instruction
        self.action_plan = collections.deque()

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
                "domain_id": 0,
                "steps": 10,
            }
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                action_seq = np.array(result["action"], dtype=np.float32)
                self.action_plan.extend(action_seq.tolist())
            except Exception as e:
                print(f"[Client] Request failed: {e}")
                return np.zeros_like(self.proprio)

        action_pred = np.array(self.action_plan.popleft(), dtype=np.float32)
        # Update proprio memory
        self.proprio[:10] = action_pred[:10]

        # Postprocess 6D rotation -> Euler xyz + gripper binary
        action_final = np.concatenate([
            action_pred[:3],
            rotate6D_to_euler_xyz(action_pred[3:9]) + np.array([0, math.pi / 2, 0]),
            np.array([1.0 if action_pred[9] < 0.91 else -1.0])
        ])
        return action_final

# ======================================================
# === WidowX evaluation routine ========================
# ======================================================
def evaluate_policy_widowx(client, output_dir: str, proc_id: int, max_steps: int = 1200):
    """
    Evaluate the XVLA policy on multiple WidowX tasks via simpler_env.
    Includes user-friendly logs, timing, and error recovery.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "widowx_results.txt")

    task = "widowx_stack_cube"

    summary = []
    start_time_total = time.time()

    print("\n" + "=" * 70)
    print(f"🧩 Eval Task: {task} | Proc ID: {proc_id}")
    print("=" * 70)

    try:
        images = []
        env = simpler_env.make(task)
        obs, _ = env.reset(options={"obj_init_options": {"episode_id": proc_id}})
        instruction = env.get_language_instruction()

        # Compute EE pose wrt base
        ee_pose_wrt_base = Pose(
            p=obs["agent"]["base_pose"][:3],
            q=obs["agent"]["base_pose"][3:]
        ).inv() * Pose(
            p=obs["extra"]["tcp_pose"][:3],
            q=obs["extra"]["tcp_pose"][3:]
        )

        # Compose proprio
        proprio = torch.from_numpy(np.concatenate(
            [ee_pose_wrt_base.p, np.array([1, 0, 0, 1, 0, 0, 0])]
        )).to(dtype=torch.float32)
        proprio = torch.cat([proprio, torch.zeros_like(proprio)], dim=-1).numpy()

        # Reset XVLA client
        client.reset(proprio, instruction)

        # === Run environment loop ===
        task_start = time.time()
        for step_idx in range(max_steps):
            image = get_image_from_maniskill2_obs_dict(env, obs)
            action = client.step(image)
            obs, reward, done, _, _ = env.step(action)
            images.append(image.copy())

            if done:
                print(f"✅ Task {task} completed in {step_idx+1} steps (suc={done})")
                break
        # === Save video & log ===
        duration = time.time() - task_start
        out_video = os.path.join(output_dir, f"{task}_{proc_id}_{done:.2f}.mp4")
        write_video(out_video, images, fps=10)

        result = {
            "task": task,
            "reward": float(reward),
            "done": bool(done),
            "steps": step_idx + 1,
            "duration_sec": duration,
            "output": out_video,
            "proc_id": proc_id,
        }
        summary.append(result)

        with open(log_path, "a+") as f:
            f.write(json.dumps(result) + "\n")

        print(f"🎥 Saved video to {out_video}")
        print(f"🕒 Task duration: {duration:.1f}s")

    except Exception as e:
        print(f"❌ Error during task {task}: {e}")
        with open(log_path, "a+") as f:
            f.write(json.dumps({
                "task": task,
                "proc_id": proc_id,
                "error": str(e)
            }) + "\n")

    # === Summary ===
    total_time = time.time() - start_time_total
    print("\n" + "=" * 70)
    print(f"🏁 Evaluation finished for proc {proc_id}")
    print(f"⏱️  Total elapsed time: {total_time/60:.2f} min")
    print(f"📊 Results written to: {log_path}")
    print("=" * 70)

# ======================================================
# === Entry ============================================
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("XVLA WidowX Evaluation Client")
    parser.add_argument("--connection_info", type=str, default=None,
                        help="Path to server info.json written by XVLA server")
    parser.add_argument("--server_ip", type=str, default=None,
                        help="Manual server IP (if not using connection_info)")
    parser.add_argument("--server_port", type=int, default=None,
                        help="Manual server port (if not using connection_info)")
    parser.add_argument("--output_dir", type=str, default="logs/",
                        help="Directory for saving evaluation videos and logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    
    print("🎯 Starting WidowX policy evaluation...")
    print(f"📁 Results and videos will be saved to: {os.path.abspath(args.output_dir)}")

    for proc_id in range(24):
        print(f"\n--- 🧩 Starting evaluation process {proc_id}/24 ---")
        try:
            evaluate_policy_widowx(client, args.output_dir, proc_id)
        except KeyboardInterrupt:
            print("🛑 Interrupted by user. Exiting gracefully...")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️ Process {proc_id} failed with error: {e}")
            continue

    print("\n✅ All evaluations completed successfully!")
    print(f"🎥 Check your videos and logs under: {os.path.abspath(args.output_dir)}")