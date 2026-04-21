# 🧪 Evaluation on RoboTwin-2.0

We evaluate **X-VLA** on the **RoboTwin-2.0** benchmark to assess its ability to handle bimanual tabletop manipulation with multiple object sets, shifting layouts, and varied visual randomness.

---

## 1️⃣ Environment Setup

Follow the official instructions from the original RoboTwin-2.0 repository:  
👉 [https://robotwin-platform.github.io/doc/usage/index.html](https://robotwin-platform.github.io/doc/usage/index.html)

No additional modifications are required for X-VLA evaluation.

---

## 2️⃣ Start the X-VLA Server

Run the X-VLA model as an inference server (in a clean environment to avoid dependency conflicts):

```bash
conda activate X-VLA
python -m deploy --model_path 2toINF/X-VLA-RoboTwin2
```
---

## 3️⃣ Run the Client Evaluation
Add the absolute path of your RoboTwin repository at line 4 of `X-VLA/evaluation/robotwin-2.0/client.py`:

```
robowin_root = Path("/home/dodo/fyc/RoboTwin") # <- Add your path
```

Launch the RoboTwin-2.0 evaluation client to connect to your X-VLA server:

```bash
cd evaluation/robotwin-2.0
bash eval_robotwin.sh
```
You can configure custome evaluation in `eval_robotwin.sh`, such as log directry, server port number, number of episodes evaluated, task config, etc.

The client will stream observations (images, proprioception, and language) to the X-VLA model, receive predicted actions, and execute them within the RoboTwin-2.0 environment.

---

## 📊 Results (Using RoboTwin-2.0 Leaderboard Settings)

|    **Settings**   |   Easy  |   Hard  |
| :--------------------: | :--: | :--: |
|     **Success (%)**    | 70.0 | 39.0 |



