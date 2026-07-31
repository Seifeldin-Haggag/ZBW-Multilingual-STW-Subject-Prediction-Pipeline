
#  ZBW Multilingual STW Subject Prediction & Universal Model Framework

An end-to-end, data-centric deep learning repository for automated multi-label subject indexing of economic abstracts against the **Standard Thesaurus for Economics (STW)** vocabulary.

This repository features baseline models (**XLM-RoBERTa** and **English RoBERTa**) enhanced with custom **Multi-Label Focal Loss** to combat severe class imbalance. The architecture is fully modular, allowing you to dynamically fine-tune **ANY open-source language model on the internet** (such as **Qwen-2.5**, **LLaMA-3.2**, or **Mistral**) directly from Hugging Face via a single terminal command.

---

##  Repository Structure & Key Scripts

```text
multilingual-stw-nlp-main/
├── data/
│   ├── raw/
│   │   ├── econbiz_stw.json          # Raw JSON dataset containing abstracts & subjects
│   │   ├── econbiz_stw.tsv           # Original TSV export
│   │   └── stw-en.tsv                # STW English label definitions
│   ├── filtered/                     # Intermediate filtered data assets
│   └── processed/
│       ├── train.json                # Preprocessed training dataset (80%)
│       ├── val.json                  # Preprocessed validation dataset (20%)
│       ├── label_encoder.pkl         # Serialized MultiLabelBinarizer metadata
│       ├── pos_weights.pt            # Pre-calculated class imbalance tensors
│       └── xlm_roberta_trained_model/ # Saved fine-tuned transformer weights
├── src/
│   ├── conversion/
│   │   └── convert_tsv_to_json.py    # TSV-to-JSON formatting utility
│   ├── data_preparation/
│   │   ├── analyze_label_distribution.py # Class frequency inspection tool
│   │   ├── data_analysis.py          # Data volume and length diagnostics
│   │   └── preprocessing.py          # Noise filtering & vocabulary binarization
│   ├── evaluation/
│   │   ├── evaluate_multi_file.py    # Cross-dataset evaluation script
│   │   └── evaluate_threshold.py     # Probability gate tuning (e.g., T = 0.40 / 0.50)
│   ├── filters/
│   │   ├── filter_english_stw.py     # Language filtering utility
│   │   └── filter_subject_matching.py# Vocabulary normalization and cleaning
│   ├── inference/
│   │   ├── inference.py              # Single-document prediction API
│   │   └── interactive_inference.py  # Interactive CLI console testing
│   └── modeling_multilingual/
│       ├── cross_validation/
│       │   └── cross_validate.py     # K-Fold validation framework
│       └── xlm_roberta_train_model.py# Primary training engine with dynamic model support
├── requirements.txt                  # Python dependency requirements
├── Run_Instructions.md               # Quick execution cheat sheet
└── README.md                         # Primary project documentation

```

---

##  Quickstart Execution Pipeline

### 1. Data Preparation & Preprocessing

Filter out noise terms, process vocabulary classes, and generate `train.json`, `val.json`, and `label_encoder.pkl`:

```bash
python src/data_preparation/preprocessing.py

```

### 2. Model Training (Default XLM-RoBERTa Baseline)

Train the default baseline using custom Multi-Label Focal Loss:

```bash
python src/modeling_multilingual/xlm_roberta_train_model.py

```

### 3. Model Evaluation & Threshold Tuning

Evaluate model performance and discover optimal decision gates:

```bash
python src/evaluation/evaluate_threshold.py

```

---

##  Dynamic Model Swapping: Training ANY Model from the Internet

This repository is built with **Hugging Face `Auto` classes**, meaning you are not limited to XLM-RoBERTa. You can train **any open-source transformer model on Hugging Face** by passing its model ID flag during execution.

### How to Run Any Hugging Face Model:

* **Train on Qwen 2.5 (1.5B):**
```bash
python src/modeling_multilingual/xlm_roberta_train_model.py --model_id Qwen/Qwen2.5-1.5B

```


* **Train on LLaMA 3.2 (3B):**
```bash
python src/modeling_multilingual/xlm_roberta_train_model.py --model_id meta-llama/Llama-3.2-3B

```


* **Train on Mistral (7B):**
```bash
python src/modeling_multilingual/xlm_roberta_train_model.py --model_id mistralai/Mistral-7B-v0.1

```



---

##  Core Loss Engine: Multi-Label Focal Loss

To address extreme multi-label class imbalance, the training engine overrides standard Binary Cross Entropy (BCE) with a **Multi-Label Focal Loss** implementation:

$$\text{Focal Loss} = - \alpha (1 - p_t)^\gamma \log(p_t)$$

* **$\gamma = 2.0$ (Focusing Parameter):** Suppresses loss contributions from easily predicted background negative classes, forcing gradients to update on hard-to-classify economic tags.
* **$\alpha = 0.25$ (Alpha Balancing Factor):** Balances positive target classes against unlabelled space.

---

##  Advanced Architecture Modifications

### 1. Removing the 100-Label Cap (Unlocking Infinite STW Classes)

By default, `preprocessing.py` isolates the top 100 target categories. To train across the **entire dataset vocabulary** (hundreds or thousands of STW classes):

#### A. Modify `src/data_preparation/preprocessing.py`

Replace `.most_common(100)` with full vocabulary extraction:

```python
# OLD (Restricted to Top 100):
# top_100_names = set(label for label, _ in label_counts.most_common(100))

# NEW (Unlocks ALL STW Classes):
MIN_LABEL_FREQUENCY = 5  # Filter out ultra-rare tags appearing fewer than 5 times
all_valid_names = set(label for label, count in label_counts.items() if count >= MIN_LABEL_FREQUENCY)

# Fit MultiLabelBinarizer on the entire vocabulary space
mlb = MultiLabelBinarizer(classes=sorted(list(all_valid_names)))

```

#### B. Dynamic Initialization in `train_model.py`

The training script automatically detects the updated label dimension from `label_encoder.pkl`:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(mlb.classes_), # Dynamically adjusts to 500, 1000, or unlimited classes
    problem_type="multi_label_classification"
)

```

---

### 2. Adjusting Data Diet & Frequency Controls (Full Power Mode)

To run the model at **full capacity** across the entire dataset without clipping data volume:

* **Remove Sample Limits:** Ensure no maximum sample cap (e.g., `MAX_SAMPLES_PER_TAG`) is clipping instances in `preprocessing.py`.
* **Increase Max Sequence Length:** In `xlm_roberta_train_model.py`, increase `MAX_LENGTH` from `256` to `384` or `512` to capture complete abstract contexts.

---

### 3. Parameter-Efficient Fine-Tuning (PEFT / LoRA for LLMs)

When training larger models (3B+ parameters) on local GPUs, enable **LoRA** to fine-tune adapter weights without running out of VRAM:

```python
from peft import LoraConfig, get_peft_model, TaskType

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,                         # Rank dimension
    lora_alpha=32,                # Scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Target attention projections
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

```

---

##  License & Attribution

Data sourced from the **ZBW – Leibniz Information Centre for Economics**. Built using PyTorch, Hugging Face Transformers, and Scikit-Learn.

```

```
