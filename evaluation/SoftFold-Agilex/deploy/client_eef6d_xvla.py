#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import torch
import numpy as np
import os
import time
import argparse
import collections
from collections import deque

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, Pose, PoseStamped
from sensor_msgs.msg import JointState, Image
from piper_msgs.msg import PosCmd
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
import math
import threading

import json_numpy
import requests
import sys
sys.path.append("./")

from deploy.utils.rotation import eef_6d, eef_quat, abs_6d_2_abs_euler
from deploy.utils.rosoperator import RosOperator

task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None


class ClientModel():
    def __init__(self,
                 host,
                 port):

        self.url = f"http://{host}:{port}/act"
        self.reset()
        self.pred_proprio = None
        
    def reset(self):
        """
        This is called at the start of one episode to clear out the cached action chunks in last run.
        """
        # currently, we dont use historical observation
        self.action_plan = collections.deque()
        return None
    
    def set_proprio(self, proprio):
        """
        Here we maintain an self.pred_proprio, which is initilized by the sensor at first, and then is calculated rather than being recorded
        to enhance the smoothness of robot execution.
        """
        if self.pred_proprio is None:
            self.pred_proprio = proprio

    def rel_to_abs(self, action_list): # [B, 20]
        """
        This is a function to transform the relative xyz positioin to absolution xyz position (optionally used)
        """
        action = np.asarray(action_list)
        left_xyz_rel = action[:, :3]  # [B, 3]
        left_6d_abs = action[:, 3:9]
        left_gripper = action[:, 9]
        right_xyz_rel = action[:, 10:13]
        right_6d_abs = action[:, 13:19]
        right_gripper = action[:, 19]
        
        # base xyz
        proprio = self.pred_proprio
        left_xyz_base = proprio[:3]  # (3, )
        right_xyz_base = proprio[10:13]
        
        left_xyz_abs = left_xyz_rel + left_xyz_base
        right_xyz_abs = right_xyz_rel + right_xyz_base
        
        action_deploy = np.concatenate([left_xyz_abs, left_6d_abs, left_gripper[:, None], right_xyz_abs, right_6d_abs, right_gripper[:, None]], axis=-1)
        return action_deploy

    def step(self, obs, args):
        """
        Args:
            obs: (dict) environment observations
            - images/cam_high
            - images/cam_left_wrist
            - eef_6d
            
        Returns:
            action: (np.array) predicted action
        """
        if not self.action_plan:
            main_view = obs['images']['cam_high']   #  np.ndarray with shape (480, 640, 3)
            left_wrist_view = obs['images']['cam_left_wrist']   # np.ndarray with shape (480, 640, 3) 
            right_wrist_view = obs['images']['cam_right_wrist']   # np.ndarray with shape (480, 640, 3) 
            # self.pred_proprio = obs['eef_6d'].astype(np.float32)
            proprio = self.pred_proprio.astype(np.float32)
            language_instruction = 'flatten the cloth and then fold it, then place it to the right side of you'
            
            query = {"proprio": json_numpy.dumps(proprio),
                    "image0": json_numpy.dumps(main_view),
                    "image1": json_numpy.dumps(left_wrist_view),
                    "image2": json_numpy.dumps(right_wrist_view),
                    "language_instruction": language_instruction,
                    "steps": 10,  # ode solver step for flow matching
                    "domain_id": 5}

            response = requests.post(self.url, json=query)
            action = response.json()['action']
            action_deploy = action
            
            # (optional) transform the rel xyz pos to abs xyz pos
            # action_deploy = self.rel_to_abs(action)   # dont use this if the action is already absolute action
            
            
            # (optional) add function here to further enhance deployment smoothness
            # action_origin = np.asarray(action_deploy[:args.chunk_size])
            # action_deploy = self.smooth_transition_linear(proprio, action_origin, K=5, ease=True)
        
            self.action_plan.extend(action_deploy[:args.chunk_size])
            self.pred_proprio = np.asarray(self.action_plan[-1]).astype(np.float32)
        print("Action chunk remains:", len(self.action_plan))
        
        # binary gripper
        action_predict = np.array(self.action_plan.popleft())
        action_predict[-1] = -0.0054700000174343586 if action_predict[-1] > 0.5 else 0.06557999688386917
        action_predict[9] = -0.0054700000174343586 if action_predict[9] > 0.5 else 0.06557999688386917
        return action_predict

def get_action(args, config, ros_operator, policy, t, pre_action):
    print_flag = True

    rate = rospy.Rate(args.publish_rate)
    while True and not rospy.is_shutdown():
        # skip the ros query if the action plan is not empty
        if len(policy.action_plan) > 0:
            # print("using exiting chunk with left chunk len:", len(policy.action_plan))
            start_time = time.time()
            all_actions = policy.step(None, args)
            all_actions = abs_6d_2_abs_euler(all_actions)
        
            end_time = time.time()
            # print("model cost time:", end_time - start_time)
            inference_lock.acquire()
            inference_actions = all_actions
            if pre_action is None:
                pre_action = obs['eef_quat']

            inference_timestep = t
            inference_lock.release()
            return inference_actions #, _, _
            
        # query the camera view if the action plan is empty
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, puppet_arm_left_pose, puppet_arm_right_pose, robot_base, status) = result
        obs = collections.OrderedDict()
        image_dict = dict()

        image_dict[config['camera_names'][0]] = img_front
        image_dict[config['camera_names'][1]] = img_left
        image_dict[config['camera_names'][2]] = img_right


        obs['images'] = image_dict

        if args.use_depth_image:
            image_depth_dict = dict()
            image_depth_dict[config['camera_names'][0]] = img_front_depth
            image_depth_dict[config['camera_names'][1]] = img_left_depth
            image_depth_dict[config['camera_names'][2]] = img_right_depth
            obs['images_depth'] = image_depth_dict
            
        obs['eef_quat'] = eef_quat(puppet_arm_left, puppet_arm_right, puppet_arm_left_pose, puppet_arm_right_pose)
        obs['eef_6d'] = eef_6d(puppet_arm_left, puppet_arm_right, puppet_arm_left_pose, puppet_arm_right_pose)
        obs['qpos'] = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        obs['qvel'] = np.concatenate(
            (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
        obs['effort'] = np.concatenate(
            (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        if args.use_robot_base:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]

        start_time = time.time()
        policy.set_proprio(obs['eef_6d'])
        all_actions = policy.step(obs, args)
        all_actions = abs_6d_2_abs_euler(all_actions) 
        
        end_time = time.time()
        print("model cost time: ", end_time -start_time)
        inference_lock.acquire()
        inference_actions = all_actions
        if pre_action is None:
            pre_action = obs['eef_quat']

        inference_timestep = t
        inference_lock.release()
        return inference_actions #, obs['eef_6d'], status


def model_inference(args, config, ros_operator, save_episode=True):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    
    policy = ClientModel(args.host, args.port)
    max_publish_step = config['episode_len']

    # zero position
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
    
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Enter any key to continue :")
    
    action = None
    # inference
    start_time = time.time()
    count = 0
    
    with torch.inference_mode():
        policy.reset()

        t = 0
        rate = rospy.Rate(args.publish_rate)
        
        while t < max_publish_step and not rospy.is_shutdown():
            pre_action = action
            step_start_time = time.time()
            action = get_action(args, config, ros_operator, policy, t, pre_action)
            duration = time.time() - start_time
            count += 1
            print("avg Hz:", count/duration)
            
            left_action = action[:7]
            right_action = action[7:14]
            ros_operator.eef_arm_publish(left_action, right_action)  # eef publish

            t += 1
            rate.sleep()
            
            
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=1000000000000, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--control_arm_left_pose_topic', action='store', type=str, help='control_arm_left_pose_topic',
                        default='/control/end_pose_left', required=False)
    parser.add_argument('--control_arm_right_pose_topic', action='store', type=str, help='control_arm_right_pose_topic',
                        default='/control/end_pose_right', required=False)

    # topic name of arm end pose
    parser.add_argument('--puppet_arm_left_pose_topic', action='store', type=str, help='puppet_arm_left_pose_topic',
                        default='/puppet/end_pose_left', required=False)
    parser.add_argument('--puppet_arm_right_pose_topic', action='store', type=str, help='puppet_arm_right_pose_topic',
                        default='/puppet/end_pose_right', required=False)
 
    # topic name of arm status
    parser.add_argument('--status_topic', action='store', type=str, help='puppet_arm_left_pose_topic',
                        default='/puppet/arm_status', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=15, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=30, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    parser.add_argument('--host', action='store', type=str, help='rel eef or abs eef',
                        default="0.0.0.0", required=False)    
    parser.add_argument('--port', action='store', type=str, help='rel eef or abs eef',
                        default="8000", required=False)    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = {
        'episode_len': args.max_publish_step,
        'camera_names': task_config['camera_names'],
    }
    model_inference(args, config, ros_operator, save_episode=True)


if __name__ == '__main__':
    main()