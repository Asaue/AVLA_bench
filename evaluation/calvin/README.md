# 🧪 Evaluation on CALVIN (ABC→D)

We evaluate **X-VLA** on the **CALVIN ABC→D** benchmark to assess its ability to generalize across long-horizon, multi-stage, language-conditioned manipulation tasks.

---

## 1️⃣ Environment Setup

Follow the official instructions from the original CALVIN repository:  
👉 [https://github.com/mees/calvin](https://github.com/mees/calvin)

No additional modifications are required for X-VLA evaluation.

---

## 2️⃣ Start the X-VLA Server

Run the X-VLA model as an inference server (in a clean environment to avoid dependency conflicts):

```bash
conda activate X-VLA
python -m x_vla.deploy \
  --model_path  2toINF/X-VLA-Calvin-ABC_D\
  --host 0.0.0.0 \
  --port 8000
```
---

## 3️⃣ Run the Client Evaluation

Launch the CALVIN evaluation client to connect to your X-VLA server:

```bash
cd evaluation/calvin
python client.py --server_ip 127.0.0.1 --server_port 8000
```

The client will stream observations (images, proprioception, and language) to the X-VLA model, receive predicted actions, and execute them within the CALVIN environment.

---

## 📊 Results (Example)

|    **Stage (ABC→D)**   |   1  |   2  |   3  |   4  |     5    |
| :--------------------: | :--: | :--: | :--: | :--: | :------: |
|     **Success (%)**    | 97.1 | 92.6 | 88.5 | 84.4 |   78.8   |
| **Final CALVIN Score** |   —  |   —  |   —  |   —  | **4.43** |

> The per-stage values represent **success rates**, and the final value is the **official CALVIN score**.

