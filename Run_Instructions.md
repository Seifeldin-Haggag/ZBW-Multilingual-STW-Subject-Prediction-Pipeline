# RoBERTa Fine-Tuning for Multi-Label Subject Classification

## Objective

Given an academic abstract, predict which subjects (e.g., "Machine Learning", "Energy", "Economics") it belongs to.

---

### 1. 🔄 Preprocess Dataset

**Script:** `src/preprocessing.py`

* Loads raw data `english_stw_filtered.json`
* Extracts abstracts and subject labels
* Encodes labels with `MultiLabelBinarizer`
* Splits data into `train.json`, `val.json` and `test.json`

**Run:**

```bash
python3 src/preprocessing.py
```
---

### 3. Fine-Tune RoBERTa

**Script:** `src/train_roberta_filtered.py`

* Loads `train.json` and `val.json`
* Filters to top 50 most frequent labels
* Uses weighted `BCEWithLogitsLoss` to combat class imbalance
* Fine-tunes full RoBERTa model
* Saves model + tokenizer to `roberta_trained_model/`

**Run:**

```bash
python3 src/train_roberta_filtered.py
```

---

### 4. Evaluate Model

**Script:** `src/evaluate_threshold.py`

* Loads fine-tuned model + tokenizer
* Evaluates predictions on `val.json`
* Computes F1/Precision/Recall across thresholds 0.2–0.8

**Run:**

```bash
python3 src/evaluate_threshold.py
```

---

## Custom Trainer

**File:** `train_roberta_filtered.py`

* Subclasses Hugging Face `Trainer`
* Uses `pos_weight` with `BCEWithLogitsLoss` for rare label handling

---

## Setup & Installation

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

If `skmultilearn` fails:

```bash
pip3 install git+https://github.com/scikit-multilearn/scikit-multilearn.git
```

---

## Full Pipeline Commands

```bash
source .venv/bin/activate
python3 src/preprocessing.py
python3 src/train_roberta_filtered.py
python3 src/evaluate_threshold.py
```
---