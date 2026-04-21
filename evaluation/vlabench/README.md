# Evaluation on VLABench

We evaluate **X-VLA** on the VLABench benchmark, which consists of four subtasks: **track_1_in_distribution**, **track_2_cross_category**, **track_3_common_sense**, and **track_4_semantic_instruction**. We report the average progress score to assess the policy’s skill learning ability, generalization ability, and overall task-related competence.

---

## 1️⃣ Environment Setup

Set up VLABench following the [official instructions](https://github.com/OpenMOSS/VLABench).

```sh
# Prepare conda environment
cd evaluation/vlabench
conda create -n vlabench python=3.10
conda activate vlabench
git clone https://github.com/OpenMOSS/VLABench.git
cd VLABench
pip install -r requirements.txt
pip install -e .

# Download the assets
python scripts/download_assets.py
```
---

## 2️⃣ Start the X-VLA Server

Run the X-VLA model as an inference server (in a clean environment to avoid dependency conflicts):
```bash
cd X-VLA
conda activate X-VLA
python -m deploy \
  --model_path 2toINF/X-VLA-VLABench \
  --port 8000
```
---

## 3️⃣ Run the Client Evaluation

Launch the VLABench evaluation client to connect to your X-VLA server:
```bash
cd evaluation/vlabench
conda activate vlabench
python vlabench_client.py \
    --eval-track track_1_in_distribution track_2_cross_category track_3_common_sense track_4_semantic_instruction \
    --metrics "success_rate" "intention_score" "progress_score" \
    --host 0.0.0.0 \
    --port 8000 
```

