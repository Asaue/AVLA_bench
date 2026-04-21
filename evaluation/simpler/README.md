# 🧪 Evaluation on SIMPLER Benchmark

We evaluate **X-VLA** on the **SIMPLER benchmark**, covering both **WidowX** and **Google Robot** embodiments.  
The evaluation follows [SimplerEnv](https://github.com/255isWhite/SimplerEnv), with **minor environment modifications** to support **absolute end-effector (EE) control** — details can be found in our GitHub commit history.
---

## 🚀 Quick Evaluation Steps

### 1️⃣ Environment Setup
```bash
# Make sure X-VLA has been correctly installed before this
conda activate XVLA
git clone https://github.com/255isWhite/SimplerEnv.git --recurse-submodules
realpath SimplerEnv # copy this path as simpler_env_path, it will be used for Google Robot evaluation
cd SimplerEnv/ManiSkill2_real2sim
pip install -e .
cd ..
pip install -e .
```

*(Ensure MuJoCo and EGL rendering are correctly configured.)*

---

### 2️⃣ Launch X-VLA Server (The **first** terminal for server)

```bash
cd X-VLA

# for WidowX
python deploy.py \
    --model_path 2toINF/X-VLA-WidowX \
    --host 0.0.0.0 \
    --port 8000

# for Goole Robot
python deploy.py \
    --model_path 2toINF/X-VLA-Google-Robot \
    --host 0.0.0.0 \
    --port 8000
```
*(Change the port accordingly.)*

---

### 3️⃣ Run Client Evaluation (The **second** terminal for client)

```bash
# for WidowX
cd X-VLA/evaluation/simpler/WidowX
# choose task from client_spoon/client_carrot/client_blocks/client_eggplant
python client_spoon.py --server_ip 127.0.0.1 --server_port 8000

# for Goole Robot
# choose settings from google-VM/google-VA
cd X-VLA/evaluation/simpler/google-VM
# choose task from client_coke_can/client_move_near/client_open_close/client_place_in
export SIMPLER_DIR=path/to/SimplerEnv
python client_coke_can.py --server_ip 127.0.0.1 --server_port 8000
```

This client:

* Connects to the X-VLA inference server
* Sends proprioceptive + visual observations
* Executes predicted action sequences (ΔEE6D or AbsEE depending on env setup)
* Logs success metrics automatically

---

## 📊 Results on Simpler Benchmark

### WidowX (Visual Matching)

| **Spoon** | **Carrot** | **Blocks** | **Eggplant** | **Avg.** |
| :-------: | :--------: | :--------: | :-----------: | :------: |
|**100**|**91.7**|**95.8**| **95.8**  | **95.8** |

### Google RT1 (Visual Matching)

| **Coke** | **Near** | **Open** | **Place** | **Avg.** |
| :------: | :------: | :------: | :-----: | :------: |
| **98.3** | **97.5** | **74.5** | **63.8** | **83.5** |


### Google RT1 (Visual Aggregation)

| **Coke** | **Near** | **Open** | **Place** | **Avg.** |
| :------: | :------: | :------: | :-----: | :------: |
| **87.3** | **84.0** | **64.2** | **70.3** | **76.4** |



---
*The SIMPLER benchmark currently has uncontrollable randomness, so results may vary even with the same settings. We are fixing this, and our reported numbers are taken from the best rollout.*