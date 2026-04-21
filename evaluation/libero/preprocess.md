# Preprocessing the dataset

We follow [OpenVLA](https://github.com/openvla/openvla) to regenerate the Libero dataset with higher image resolution and filter out failed rollouts. Specifically, run the script `openvla/experiments/robot/libero/regenerate_libero_dataset.py`.

# Action Transformation

### Retrieving Absolute Actions
The original actions in the dataset are relative EEF actions. In `env.step()`, the relative actions are first converted into absolute EEF positions and orientations before being fed into the controller. These absolute actions are then stored in controller.goal_pos and controller.goal_ori. Therefore, we can simply replay the training data to retrieve them. For this step, you can refer to our [implementation](https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/rel2abs.py).

### Converting Actions into EEF-6D Format
After retrieving the absolute EEF actions (xyz (3) + axis-angle (3) + gripper (1)), we convert them into absolute EEF-6D actions (xyz (3) + eef6d (6) + gripper (1)) by transforming the axis-angle representation into the 6D rotation representation. For this step, please refer to our [implementation](https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/libero_client.py#L126), specifically the `AxisAngle_to_Rotate6D()` function in the `LiberoAbsActionProcessor` class.
