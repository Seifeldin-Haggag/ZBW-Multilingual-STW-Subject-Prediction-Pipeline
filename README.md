#  ZBW Multilingual STW Subject Prediction & Universal Model Framework

An end-to-end, data-centric deep learning repository for automated multi-label subject indexing of economic abstracts against the **Standard Thesaurus for Economics (STW)** vocabulary.

This repository features baseline models (**XLM-RoBERTa** and **English RoBERTa**) enhanced with custom **Multi-Label Focal Loss** to combat severe class imbalance. The architecture is fully model-agnostic, allowing you to dynamically fine-tune **ANY open-source language model on the internet** (such as **Qwen-2.5**, **LLaMA-3.2**, or **Mistral**) directly from Hugging Face via a simple command-line argument.

---

##  Repository Structure & Key Scripts

```text
multilingual-stw-nlp-main/
├── data/
│   ├── raw/
│   │   ├── econbiz_stw.json          # Raw JSON dataset containing abstracts & subjects
│   │   ├── econbiz_stw.tsv           # Original TSV export
│   │   └── stw-en.tsv                # STW English label definitions
│   ├── filtered/                     # Intermediate vocabulary-filtered assets
│   │   └── english_stw_filtered.json # Vocabulary intersected abstract dataset
│   └── processed/
│       ├── train.json                # Preprocessed training dataset (80%)
│       ├── val.json                  # Preprocessed validation dataset (20%)
│       ├── label_encoder.pkl         # Serialized MultiLabelBinarizer metadata
│       └── trained_model_.../        # Saved fine-tuned transformer weights & tokenizers
├── src/
│   ├── conversion/
│   │   └── convert_tsv_to_json.py    # TSV-to-JSON formatting utility
│   ├── data_preparation/
│   │   ├── analyze_label_distribution.py # Class frequency inspection tool
│   │   ├── data_analysis.py          # Data volume and length diagnostics
│   │   └── preprocessing.py          # Full-vocabulary class extraction & binarization
│   ├── evaluation/
│   │   ├── evaluate_multi_file.py    # Cross-dataset evaluation script
│   │   └── evaluate_threshold.py     # Probability gate tuning (e.g., T = 0.40 / 0.50)
│   ├── filters/
│   │   ├── filter_english_stw.py     # Vocabulary matching & intersection filter
│   │   └── filter_subject_matching.py# Vocabulary normalization utility
│   ├── inference/
│   │   ├── inference.py              # Single-document prediction API
│   │   └── interactive_inference.py  # Interactive CLI console testing
│   └── models/
│       └── train.py                  # Universal training engine with Focal Loss & CLI support
├── requirements.txt                  # Dependencies list
└── README.md                         # Primary project documentation

```

---

##  Step-by-Step Execution Guide

Follow these steps in order to process the raw dataset, generate binarized label matrices, train your target model, and evaluate predictions.

### Step 1: Environment Setup

Ensure all required Python packages are installed:

```bash
pip install -r requirements.txt

```

### Step 2: Vocabulary Intersection Filtering

Intersects your raw JSON records with the official STW vocabulary definitions to ensure clean target terms:

```bash
python src/filters/filter_english_stw.py

```

* **Output:** Creates `data/filtered/english_stw_filtered.json`.

### Step 3: Dataset Preprocessing & Full-Vocabulary Binarization

Extract all valid target labels (including country tags like Germany, USA, etc.), fit the `MultiLabelBinarizer`, and generate train/val splits:

```bash
python src/data_preparation/preprocessing.py

```

* **Output:** Creates `data/processed/train.json`, `data/processed/val.json`, and `data/processed/label_encoder.pkl`.

### Step 4: Model Training

Run model fine-tuning using the universal training engine (defaults to `xlm-roberta-base`):

```bash
python src/models/train.py

```

### Step 5: Model Evaluation & Threshold Tuning

Discover optimal probability decision thresholds (e.g., $T = 0.40$ vs. $T = 0.50$) across macro-F1, Precision, and Recall:

```bash
python src/evaluation/evaluate_threshold.py

```

### Step 6: Interactive Inference

Test subject tag predictions on custom text inputs directly in your terminal:

```bash
python src/inference/interactive_inference.py

```

---

##  Dynamic Model Swapping: Fine-Tuning ANY Model on the Internet

Because `src/models/train.py` utilizes Hugging Face `Auto` classes (`AutoTokenizer` and `AutoModelForSequenceClassification`) alongside a dynamic `--model_id` parameter, you can train **any open-source transformer model on Hugging Face**:

```bash
# Fine-tune XLM-RoBERTa (Default Multilingual Baseline)
python src/models/train.py --model_id xlm-roberta-base

# Fine-tune English RoBERTa Baseline
python src/models/train.py --model_id roberta-base

# Fine-tune Qwen 2.5 (1.5B Parameter LLM)
python src/models/train.py --model_id Qwen/Qwen2.5-1.5B

# Fine-tune LLaMA 3.2 (3B Parameter LLM)
python src/models/train.py --model_id meta-llama/Llama-3.2-3B

```

*Outputs are saved automatically in distinct subdirectories under `data/processed/trained_model_<model_name>/`.*

---

##  Data Scaling Controls: How to Change Data Volume ("Data Diet")

Whether you are debugging code locally, testing on a small laptop GPU, or running a full-capacity production training run, you can adjust the volume of data flowing into the pipeline using two methods:

### Method 1: Subsampling the Training Set (Quick Experimentation)

To train faster on a small subset of the data without re-processing files, edit `TRAIN_SAMPLE_SIZE` inside `src/models/train.py`:

```python
# In src/models/train.py:

TRAIN_SAMPLE_SIZE = 2000  # Set to an integer to randomly train on 2,000 samples only
# TRAIN_SAMPLE_SIZE = None  # Set to None for FULL POWER MODE (uses 100% of the dataset)

```

### Method 2: Adjusting Vocabulary Frequency Thresholds (Label Space Control)

To control how many unique STW target classes the model learns, modify `MIN_LABEL_FREQUENCY` in `src/data_preparation/preprocessing.py`:

* **Full Capacity / All Classes (Default):**
```python
MIN_LABEL_FREQUENCY = 1  # Unlocks all categories including countries and concepts

```


* **Filtered Frequency Threshold:**
```python
MIN_LABEL_FREQUENCY = 5  # Restricts target categories strictly to tags appearing 5+ times

```



---

##  Core Loss Engine: Multi-Label Focal Loss

To address extreme multi-label class imbalance across thousands of potential topics, the training engine overrides standard Binary Cross Entropy (BCE) with a **Multi-Label Focal Loss** implementation:

$$\text{Focal Loss} = - \alpha (1 - p_t)^\gamma \log(p_t)$$

* **$\gamma = 2.0$ (Focusing Parameter):** Suppresses loss contributions from easy negative classes, forcing gradient updates on difficult subject boundaries.
* **$\alpha = 0.25$ (Alpha Balancing Factor):** Balances positive target classes against unlabelled space.

---

##  License & Attribution

Data sourced from the **ZBW – Leibniz Information Centre for Economics**. Built using PyTorch, Hugging Face Transformers, and Scikit-Learn.

```

```
