# 🧪 Evaluation on Agilex

We evaluate **X-VLA** on the **Agilex Aloha platform** to perform long-horizon cloth-folding tasks.
---

## 🚀 Quick Evaluation Steps

### 1️⃣ Environment Setup
If you directly buy one Agilex Aloha plaform, Agilex has installed an `aloha` environment, which includes most of the requirements. You just need to install several packages for rotation transformation and server-client communication:

``` bash
conda activate aloha
pip install json_numpy
pip install requests
pip install scipy
pip install numpy==1.24.4
```


Additionally, you should to change few lines to modify the urdf path to the one in our directory, since we manually solve the IK to avoid some bugs in original Agilex platform

1. open `~/X-VLA/evaluation/SoftFold-Agilex/Piper_ros_private-ros-noetic/src/piper/scripts/piper_start_ms_node.py`

2. change the path in `os.path.join("PATH")` at line 53 to the path of the file `evaluation/SoftFold-Agilex/Piper_ros_private-ros-noetic/src/piper_description/urdf/piper_description.urdf` contained in our repo.

---

### 2️⃣ Launch X-VLA Server

Navigate to your X-VLA's main folder, then launch the server

```bash
python deploy.py \
    --model_path 2toINF/X-VLA-SoftFold \
    --host 0.0.0.0 \
    --port 8000
```

or via Python:

```python
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained("2toINF/X-VLA-SoftFold", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("2toINF/X-VLA-SoftFold", trust_remote_code=True)
# model = model.to("cuda") # or "cpu" according to your device type
model.run(processor, host="0.0.0.0", port=8000)
```

---

### 3️⃣ Run Client Evaluation

```bash
cd X-VLA/evaluation/SoftFold-Agilex/Piper_ros_private-ros-noetic

bash can_config.sh  # rename can
roslaunch astra_camera multi_camera.launch # launch the camera nodes

# start a new terminal
cd X-VLA/evaluation/SoftFold-Agilex/Piper_ros_private-ros-noetic
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true  # launch the robot arm nodes

# start a new terminal
cd X-VLA/evaluation/SoftFold-Agilex
python deploy/client_eef6d_xvla.py --host 0.0.0.0 --port 8000 --publish_rate 15 # 🌟 Run the evaluation script
```

This client:

* Connects to the X-VLA inference server
* Sends proprioceptive + visual observations
* Executes predicted action sequences
---
