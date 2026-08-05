#  Step-by-Step Pipeline Run Instructions

This guide provides the exact terminal command sequence to run the entire project pipeline from scratch—from raw data filtering to dynamic model training and interactive inference.

---

##  Prerequisites & Setup

1.**Open your Terminal** and navigate to the project root directory:
   ```bash
   cd /PROJECT_DIRECTORY

```

2.**Install Python Dependencies:**
```bash
pip install -r requirements.txt

```



---

##  Step 1: Vocabulary Intersection Filtering

Intersects your raw JSON dataset (`data/raw/econbiz_stw.json`) against the official STW English reference vocabulary (`data/raw/stw-en.tsv`) to remove invalid tags:

```bash
python src/filters/filter_english_stw.py

```

* **Output generated:** `data/filtered/english_stw_filtered.json`

---

##  Step 2: Full-Vocabulary Preprocessing & Encoding

Extracts all target subject labels across the entire dataset (including geographical concepts like Germany, USA, world, etc.), fits the `MultiLabelBinarizer` across the full vocabulary space, and builds an 80/20 train/validation split:

```bash
python src/data_preparation/preprocessing.py

```

* **Outputs generated:**
* `data/processed/train.json`
* `data/processed/val.json`
* `data/processed/label_encoder.pkl`



---

##  Step 3: Model Fine-Tuning (Universal Execution)

Run training using the dynamic Focal Loss engine. The framework automatically adapts to any open-source model available on Hugging Face:

### Option A: Train Default Multilingual Baseline (XLM-RoBERTa)

```bash
python src/models/train.py

```

### Option B: Train English RoBERTa Baseline

```bash
python src/models/train.py --model_id roberta-base

```

### Option C: Fine-Tune Large Language Models (Qwen 2.5 / LLaMA 3.2)

```bash
# Fine-tune Qwen 2.5 (1.5B Parameter LLM)
python src/models/train.py --model_id Qwen/Qwen2.5-1.5B

# Fine-tune LLaMA 3.2 (3B Parameter LLM)
python src/models/train.py --model_id meta-llama/Llama-3.2-3B

```

* **Outputs generated:** Checkpoints, final model weights, and tokenizers are saved under `data/processed/trained_model_<model_name>/`.

---

##  Step 4: Decision Threshold Tuning & Evaluation

Evaluate model predictions on validation data and determine the optimal probability threshold (e.g., $T = 0.40$ vs. $T = 0.50$) across Macro F1, Precision, and Recall:

```bash
python src/evaluation/evaluate_threshold.py

```

---

##  Step 5: Interactive Terminal Inference

Run real-time subject predictions on custom economic abstracts directly in your console:

```bash
python src/inference/interactive_inference.py

```

---

##  Quick Troubleshooting & Data Scaling Adjustments

* **Debugging / Quick Experimentation:**
Open `src/models/train.py` and set `TRAIN_SAMPLE_SIZE = 2000` to train quickly on a subset.
* **Full Capacity / Production Mode:**
Open `src/models/train.py` and set `TRAIN_SAMPLE_SIZE = None` to train on 100% of the dataset.

```

```
