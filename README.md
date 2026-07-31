Here is an expanded, production-ready **`README.md`** file that incorporates your entire repository structure, your custom Focal Loss setup, execution steps, and a dedicated **Advanced Architecture Modifications** section.

This new section explicitly details how to swap to LLMs, remove the 100-label cap for unlimited classes, and adjust data frequency/diet controls to maximize performance across your entire vocabulary.

---

```markdown
#  ZBW Multilingual STW Subject Prediction Pipeline

An end-to-end, data-centric deep learning repository for automated multi-label subject indexing of economic abstracts against the **Standard Thesaurus for Economics (STW)** vocabulary.

This repository features baseline models (**XLM-RoBERTa** and **English RoBERTa**) enhanced with custom **Multi-Label Focal Loss** to combat severe class imbalance. The project is structured modularly, allowing seamless adaptation to fine-tuning generative **Large Language Models (LLMs)** such as **Qwen-2.5** or **LLaMA-3**.

---

## 📂 Repository Structure & Key Scripts

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
│   │   └── preprocessing.py          # Noise filtering & Top-100 binarization
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
│       └── xlm_roberta_train_model.py# Primary XLM-RoBERTa training engine
├── requirements.txt                  # Python dependency requirements
├── Run_Instructions.md               # Quick execution cheat sheet
└── README.md                         # Primary project documentation

```

---

## Quickstart Pipeline Steps

### 1. Data Preparation & Preprocessing

Filter out noise terms, extract target STW subject classes, and generate the binarized `train.json`, `val.json`, and `label_encoder.pkl` files:

```bash
python src/data_preparation/preprocessing.py

```

### 2. Model Training

Train the multilingual baseline using XLM-RoBERTa with custom Focal Loss:

```bash
python src/modeling_multilingual/xlm_roberta_train_model.py

```

### 3. Model Evaluation & Threshold Tuning

Evaluate model performance across validation sets and discover optimal decision gates:

```bash
python src/evaluation/evaluate_threshold.py

```

### 4. Interactive Inference

Test predictions interactively in your terminal:

```bash
python src/inference/interactive_inference.py

```

---

## ⚡ Core Loss Engine: Multi-Label Focal Loss

To address extreme multi-label imbalance, the training engine overrides standard Binary Cross Entropy (BCE) with a **Multi-Label Focal Loss** implementation:

$$\text{Focal Loss} = - \alpha (1 - p_t)^\gamma \log(p_t)$$

* **$\gamma = 2.0$ (Focusing Parameter):** Suppresses loss contributions from easily predicted negative classes, forcing gradients to update on difficult economic subject boundaries.
* **$\alpha = 0.25$ (Alpha Balancing Factor):** Balances positive versus background class representations.

---

##  Advanced Architecture Modifications & Optimization Guide

To unlock the full potential of this pipeline—expanding beyond the 100-label baseline, removing data sampling limits ("data diet"), and changing model backbones—follow these technical modifications.

---

### 1. Removing the 100-Label Cap (Unlocking All Classes / Infinite Tags)

By default, `preprocessing.py` restricts training to the top 100 target categories. To allow the model to learn **all** categories present in your dataset (hundreds or thousands of STW classes):

#### A. Modify `src/data_preparation/preprocessing.py`

Locate the label extraction section and replace the `.most_common(100)` limitation with full vocabulary extraction:

```python
# OLD (Restricted to Top 100):
# top_100_names = set(label for label, _ in label_counts.most_common(100))

# NEW (Unlocks ALL STW Classes):
MIN_LABEL_FREQUENCY = 5  # Filter out ultra-rare tags appearing fewer than 5 times
all_valid_names = set(label for label, count in label_counts.items() if count >= MIN_LABEL_FREQUENCY)

# Fit MultiLabelBinarizer on the entire vocabulary space
mlb = MultiLabelBinarizer(classes=sorted(list(all_valid_names)))

```

#### B. Dynamic MultiLabelBinarizer Initialization in `train.py`

In `xlm_roberta_train_model.py`, ensure the model initialization dynamically reads the vocabulary size directly from `label_encoder.pkl` rather than hardcoding a parameter:

```python
# Automatically scales output neurons to match the size of mlb.classes_
model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(mlb.classes_), # Dynamically adjusts to 500, 1000, or unlimited classes
    problem_type="multi_label_classification"
)

```

---

### 2. Adjusting Data Diet & Frequency Controls (Full Power Mode)

The "data diet" controls the maximum instance cap per subject category to prevent dominant classes (e.g., *Germany*, *Monetary Policy*) from skewing model attention.

To run the model at **full capacity** across the entire dataset without clipping data volume:

#### A. Disable Instance Clipping in `preprocessing.py`

If your script uses a hard cap counter (e.g., `MAX_SAMPLES_PER_TAG = 3000`), comment out or remove the cap logic:

```python
# FULL POWER MODE: Include all valid abstract samples
final_texts = processed_texts
final_labels = processed_labels

```

#### B. Scale Hyperparameters for Full-Scale Training

When scaling up sequence volume and target classes, adjust hyperparameter bounds in `xlm_roberta_train_model.py`:

* **Sequence Length (`MAX_LENGTH`):** Increase from `256` to `384` or `512` to capture complete academic abstract contexts.
* **Batch Size & Gradient Accumulation:** If increasing sequence length causes GPU Out-Of-Memory (`OOM`) errors, reduce `BATCH_SIZE = 4` and set `gradient_accumulation_steps = 2` in `TrainingArguments`.

---

### 3. Upgrading to Open-Source LLMs (Qwen / LLaMA / Unsloth)

To swap out `xlm-roberta-base` for modern generative LLMs (e.g., `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`, or `meta-llama/Llama-3.2-3B`):

#### A. Update Imports in `xlm_roberta_train_model.py`

Use Hugging Face's generic `Auto` classes to handle generic transformer backbones:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "Qwen/Qwen2.5-1.5B"  # Target LLM Hugging Face ID

# Initialize generic tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# CRITICAL FOR DECODER LLMs: Map EOS token as PAD token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Initialize model with sequence classification head
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification"
)
model.config.pad_token_id = tokenizer.pad_token_id

```

#### B. Apply Parameter-Efficient Fine-Tuning (PEFT / LoRA)

For local GPUs with limited VRAM, attach **LoRA adapters** to train LLM classification heads without freezing hardware:

```bash
pip install peft

```

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
model.print_trainable_parameters()  # Trains < 1% of total parameters!

```

---

## 📄 License & Attribution

Data sourced from the **ZBW – Leibniz Information Centre for Economics**. Built using PyTorch, Hugging Face Transformers, and Scikit-Learn.

```

```