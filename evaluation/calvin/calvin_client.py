#!/usr/bin/env python3
# ------------------------------------------------------------
# X-VLA Evaluation on CALVIN (ABC→D)
# Refined client structure, compatible with X-VLA server
# ------------------------------------------------------------
import argparse
from ast import parse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
from termcolor import colored
from omegaconf import OmegaConf
import hydra
from pytorch_lightning import seed_everything
import PIL.Image as Image
import requests
import json_numpy

from calvin_agent.models.calvin_base_model import CalvinBaseModel
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_env.envs.play_table_env import get_env

# --------------------------
# Global Configs
# --------------------------
EP_LEN = 720
NUM_SEQUENCES = 1000


# --------------------------
# Utility Functions
# --------------------------
def euler_xyz_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_euler("xyz", q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6D_to_quat(v6: np.ndarray) -> np.ndarray:
    a1, a2 = v6[..., 0:5:2], v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = (a2 - proj) / np.linalg.norm(a2 - proj, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return R.from_matrix(np.stack((b1, b2, b3), axis=-1)).as_quat()


def save_video(path, frames, fps=30):
    imageio.mimsave(path, frames, fps=fps)


# --------------------------
# X-VLA Client Model Wrapper
# --------------------------
class ClientModel(CalvinBaseModel):
    def __init__(self, host, port):
        self.url = f"http://{host}:{port}/act"
        self.action_plan = []
        self.proprio = None

    def reset(self, obs):
        self.action_plan.clear()
        self.proprio = np.concatenate([
            obs["robot_obs"][:3],
            euler_xyz_to_rotate6D(obs["robot_obs"][3:6]),
            obs['robot_obs'][-1:] > 0.
        ])
        self.proprio = np.concatenate([self.proprio, np.zeros_like(self.proprio)])

    def step(self, obs, goal):
        if not self.action_plan:
            main_view = obs["rgb_obs"]["rgb_static"]
            wrist_view = obs["rgb_obs"]["rgb_gripper"]
            payload = {
                "language_instruction": goal,
                "proprio": json_numpy.dumps(self.proprio.tolist()),
                "image0": json_numpy.dumps(main_view),
                "image1": json_numpy.dumps(wrist_view),
                "domain_id": 2,
                "steps": 10
            }
            # try:
            response = requests.post(self.url, json=payload, timeout=10)
            response.raise_for_status()
            self.action_plan = response.json()["action"][:20]
            # except Exception as e:
            #     print(f"⚠️ Server request failed: {e}")
            #     return np.zeros(3), np.array([0, 0, 0, 1]), -1
        action_predict = np.array(self.action_plan.pop(0))
        self.proprio[:10] = action_predict[:10]
        return (
            action_predict[:3],
            rotate6D_to_quat(action_predict[3:9]),
            1 if action_predict[9] < 0.8 else -1
        )


# --------------------------
# Evaluation Functions
# --------------------------
def evaluate_policy(model, env, output_dir, debug=False, eval_start=0, eval_end=NUM_SEQUENCES):
    conf_dir = Path("ABC_D/validation")
    task_cfg = OmegaConf.load(conf_dir / "new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "new_playtable_validation.yaml")
    eval_sequences = list(get_sequences(NUM_SEQUENCES))

    results = []
    plans = defaultdict(list)
    for idx in range(eval_start, eval_end):
        init_state, seq = eval_sequences[idx]
        r = evaluate_sequence(env, model, task_oracle, init_state, seq, val_annotations, plans, debug, output_dir)
        results.append(r)
        with open(f"{output_dir}/log.txt", 'a+') as f:
            list_r = count_success(results)
            list_r.append(sum(list_r))
            print(f"{idx}: " ," ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(list_r)]) + "|", file=f)
    return results

def evaluate_sequence(env, model, oracle, init_state, seq, annotations, plans, debug, output_dir):
    robot_obs, scene_obs = get_env_state_for_initial_condition(init_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    model.reset(env.get_obs())
    success = 0
    for subtask in seq:
        ok, imgs, lang = rollout(env, model, oracle, subtask, annotations, plans, debug)
        save_video(f"{output_dir}/{lang}_{ok}.mp4", imgs)
        if ok:
            success += 1
        else:
            break
    return success


def rollout(env, model, oracle, subtask, annotations, plans, debug):
    obs = env.get_obs()
    lang = annotations[subtask][0].split("\n")[0].replace("\u2019", "'")
    start_info = env.get_info()
    frames = []
    for step in range(EP_LEN):
        action = model.step(obs, lang)
        obs, _, _, info = env.step(action)
        main = obs["rgb_obs"]["rgb_static"]
        wrist = np.asarray(Image.fromarray(obs["rgb_obs"]["rgb_gripper"]).resize(main.shape[:2]))
        frames.append(np.concatenate([main, wrist], axis=1))
        if step == 0:
            collect_plan(model, plans, subtask)

        if oracle.get_task_info_for_set(start_info, info, {subtask}):
            if debug:
                print(colored(f"✓ {subtask}", "green"))
            return True, frames, lang
    if debug:
        print(colored(f"✗ {subtask}", "red"))
    return False, frames, lang


# --------------------------
# Entry Point
# --------------------------
def main():
    seed_everything(0, workers=True)

    parser = argparse.ArgumentParser("XVLA CALVIN Evaluation Client")
    parser.add_argument("--connection_info", type=str, default=None,
                        help="Path to server info.json written by XVLA server")
    parser.add_argument("--server_ip", type=str, default=None,
                        help="Manual server IP (if not using connection_info)")
    parser.add_argument("--server_port", type=int, default=None,
                        help="Manual server port (if not using connection_info)")
    parser.add_argument("--output_dir", type=str, default="logs/",
                        help="Directory for saving evaluation videos and logs")
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument("--eval_end", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("🚀 [Client] Starting XVLA-CALVIN evaluation client...")

    # --------------------------------------------------
    # 1. Load server info
    # --------------------------------------------------
    if args.connection_info:
        print(f"🔍 Waiting for connection info: {args.connection_info}")
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not os.path.exists(args.connection_info):
            sys.stdout.write(f"\r{spinner[i % len(spinner)]} Waiting for server...")
            sys.stdout.flush()
            time.sleep(0.5)
            i += 1
        print("\n✅ Found connection info file!")
        with open(args.connection_info, "r") as f:
            info = json.load(f)
            host, port = info["host"], info["port"]
    else:
        if not args.server_ip or not args.server_port:
            print("❌ Must provide either --connection_info or both --server_ip and --server_port.")
            sys.exit(1)
        host, port = args.server_ip, args.server_port

    print(f"🔗 Connected to XVLA server at {host}:{port}")

    # --------------------------------------------------
    # 2. Launch environment
    # --------------------------------------------------
    print("🧩 Loading CALVIN environment...")
    env = get_env(Path("ABC_D/validation"), show_gui=False)

    # --------------------------------------------------
    # 3. Evaluate policy
    # --------------------------------------------------
    model = ClientModel(host, port)
    print(f"🎯 Starting CALVIN evaluation, saving to {args.output_dir}")
    evaluate_policy(model, env, args.output_dir, debug=False, eval_start=args.eval_start, eval_end=args.eval_end)
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()