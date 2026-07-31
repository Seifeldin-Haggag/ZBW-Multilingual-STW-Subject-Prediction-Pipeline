import os
import json
import pickle
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score

# =====================================================================
# ⚙️ GLOBAL INFERENCE PARAMS
# =====================================================================
OPTIMAL_THRESHOLD = 0.45  # The empirical sweet spot for production balance

# --- ROBUST PATH NAVIGATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed")

VAL_FILE_PATH = os.path.join(PROCESSED_DIR, "val.json")
MODEL_PATH = os.path.join(PROCESSED_DIR, "xlm_roberta_trained_model")
ENCODER_PATH = os.path.join(PROCESSED_DIR, "label_encoder.pkl")
RESULTS_PATH = os.path.join(PROCESSED_DIR, "evaluation_results_multi_file.json")

print(f"Initializing Verified Evaluation Pipeline on: {VAL_FILE_PATH}")

if not os.path.exists(VAL_FILE_PATH):
    print(f"Error: Missing validation target file at {VAL_FILE_PATH}")
    exit()

# =====================================================================
# 🚀 CORE ENGINES & HARDWARE INITIALIZATION
# =====================================================================
# Load Tokenizer & Model
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# Hardware acceleration setup
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Inference Engine backend: {device.type.upper()}")

model.to(device)
model.eval()

# Load MultiLabelBinarizer Encoder
with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)

def preprocess(text):
    return tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")

# =====================================================================
# 📊 INFERENCE LOOP
# =====================================================================
texts, actuals, predictions = [], [], []
strict_match, relaxed_match = 0, 0

with open(VAL_FILE_PATH, "r", encoding="utf-8") as f:
    items = json.load(f)

print(f"Loaded {len(items)} validation entries. Commencing GPU inferences at threshold {OPTIMAL_THRESHOLD}...")

for count, item in enumerate(items, 1):
    try:
        abstract = item["abstract"]
        true_subjects = item["subject"]
        texts.append(abstract)

        # Generate True Binary Target Vector via MLB
        true_vector = mlb.transform([true_subjects])[0].tolist()
        actuals.append(true_vector)

        # Preprocess text
        encoding = preprocess(abstract)
        
        # 🚀 FIXED: Move input tensors to the same hardware accelerator device as the model
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            logits = model(**encoding).logits
            probs = torch.sigmoid(logits).squeeze()
            
            # Map predictions dynamically using your parameterized threshold gate
            if probs.dim() == 0:  
                pred_vector = [1] if probs >= OPTIMAL_THRESHOLD else [0]
            else:
                pred_vector = (probs >= OPTIMAL_THRESHOLD).int().tolist()

        predictions.append(pred_vector)

        # Vector Matrix Symmetrical Math
        pred_arr = np.array(pred_vector, dtype=int)
        true_arr = np.array(true_vector, dtype=int)

        # 1. Strict Match Accuracy: Exact array equality
        if np.array_equal(pred_arr, true_arr):
            strict_match += 1
            
        # 2. FIXED: Relaxed Match Accuracy (Shared bits OR correct double-empty classifications)
        has_intersection = np.sum(pred_arr & true_arr) > 0
        both_empty = (np.sum(pred_arr) == 0) and (np.sum(true_arr) == 0)
        
        if has_intersection or both_empty:
            relaxed_match += 1

        if count % 2000 == 0:
            print(f"   Processed {count}/{len(items)} records...")
            
    except Exception as e:
        # Gracefully handle occasional bad data schema variations without breaking execution
        continue

if not predictions or not actuals:
    print("Fatal Error: Zero rows parsed correctly into matrix arrays. Review validation item layout schema.")
    exit()

# =====================================================================
# 🏁 GLOBAL METRIC COMPUTATION & EXPORT
# =====================================================================
predictions = np.array(predictions)
actuals = np.array(actuals)
total = len(texts)

f1 = f1_score(actuals, predictions, average="macro", zero_division=0)
precision = precision_score(actuals, predictions, average="macro", zero_division=0)
recall = recall_score(actuals, predictions, average="macro", zero_division=0)

results = {
    "evaluation_threshold": OPTIMAL_THRESHOLD,
    "f1_score": float(f1),
    "precision": float(precision),
    "recall": float(recall),
    "strict_accuracy": float(strict_match / total),
    "relaxed_accuracy": float(relaxed_match / total),
}

# Save performance data
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"\nSUCCESS! Metrics calculated cleanly at {OPTIMAL_THRESHOLD} cutoff threshold.")
print(f"Macro F1-Score:   {results['f1_score']:.4f}")
print(f"Macro Precision:  {results['precision']:.4f}")
print(f"Macro Recall:     {results['recall']:.4f}")
print("-" * 45)
print(f"Strict Accuracy:  {results['strict_accuracy']:.4f}")
print(f"Relaxed Accuracy: {results['relaxed_accuracy']:.4f}")
print("-" * 45)
print(f"Summary metrics exported smoothly to: {RESULTS_PATH}")