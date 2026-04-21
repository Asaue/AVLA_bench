import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

class FrankaMujocoHandler:
    # 告诉模型：前 9 维（3维位置 + 6维旋转）需要计算相对增量 (delta)，夹爪是绝对控制不需要 delta
    idx_for_delta = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    idx_for_mask_proprio = [] 

    def __init__(self, meta, num_views=2, **kwargs):
        self.meta = meta
        self.num_views = num_views

    def iter_episode(self, traj_idx, num_actions=10, training=True, image_aug=None, lang_aug_map=None, action_mode="ee6d", **kwargs):
        traj_path = self.meta["datalist"][traj_idx]
        csv_path = os.path.join(traj_path, "trajectory.csv")
        df = pd.read_csv(csv_path)

        instruction_text = "manipulate the object with the franka arm"

        states = []
        for _, row in df.iterrows():
            # 1. 提取 10维 的 ee6d 状态
            x, y, z = row['tool0_x'], row['tool0_y'], row['tool0_z']
            quat = [row['joint_10_qx'], row['joint_10_qy'], row['joint_10_qz'], row['joint_10_qw']]
            rot_mat = R.from_quat(quat).as_matrix() 
            rot_6d = np.concatenate([rot_mat[:, 0], rot_mat[:, 1]]) # 6D 连续旋转表示
            gripper = row['finger_joint1']
            
            # 拼接为 10维 向量
            state = np.concatenate([[x, y, z], rot_6d, [gripper]]).astype(np.float32)
            states.append(state)
        
        # 转换为 Tensor 
        all_action = torch.from_numpy(np.array(states))
        
        # 2. 核心修复：强制 Padding 到 20 维，满足官方框架的底层断言！
        pad_dim = 20 - all_action.shape[1]
        if pad_dim > 0:
            all_action = torch.cat([all_action, torch.zeros((all_action.shape[0], pad_dim))], dim=-1)

        traj_len = len(all_action)

        for i in range(traj_len):
            row = df.iloc[i]
            
            img_global_path = os.path.join(traj_path, row['image_global_path'])
            img_ee_path = os.path.join(traj_path, row['image_ee_path'])

            img_global = Image.open(img_global_path).convert("RGB")
            img_ee = Image.open(img_ee_path).convert("RGB")

            if image_aug is not None:
                img_global = image_aug(img_global)
                img_ee = image_aug(img_ee)

            # 图像张量打包
            imgs = [img_global, img_ee]
            while len(imgs) < self.num_views: 
                imgs.append(torch.zeros_like(imgs[0]))
            
            image_input = torch.stack(imgs[:self.num_views], dim=0)
            image_mask = torch.ones(self.num_views, dtype=torch.bool)

            # 3. 核心修复：切块长度必须是 num_actions + 1 (因为计算差分 delta 需要 N+1 个点)
            end_idx = min(i + num_actions + 1, traj_len)
            chunk = all_action[i : end_idx]

            # 尾部 Padding
            if end_idx - i < num_actions + 1:
                pad_len = (num_actions + 1) - (end_idx - i)
                pad = all_action[-1:].repeat(pad_len, 1)
                chunk = torch.cat([chunk, pad], dim=0)

            # 4. 严丝合缝地返回官方期望的键名，不再手动传 proprio
            yield {
                "language_instruction": instruction_text,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": chunk,
                "idx_for_delta": self.idx_for_delta,
                "idx_for_mask_proprio": self.idx_for_mask_proprio
            }