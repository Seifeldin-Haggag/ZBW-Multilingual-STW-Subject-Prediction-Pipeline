import os
import json
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# --- ROBUST ABSOLUTE PATH NAVIGATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed")
VAL_FILE = os.path.join(PROCESSED_DIR, "val.json")
ENCODER_PATH = os.path.join(PROCESSED_DIR, "label_encoder.pkl")
MODEL_PATH = os.path.join(PROCESSED_DIR, "xlm_roberta_trained_model")
MAX_LENGTH = 256
BATCH_SIZE = 8

class EvaluationDataset(Dataset):
    def __init__(self, filepath, tokenizer, mlb):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.texts = [item["abstract"] for item in data]
        
        raw_string_labels = [item["subject"] for item in data]
        self.labels = mlb.transform(raw_string_labels)
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_LENGTH, 
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

def main():
    print("Initializing Precision-Aligned Evaluation Pipeline...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Cannot find trained model at {MODEL_PATH}. Run training first!")
        return

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    
    if not os.path.exists(ENCODER_PATH):
        print(f"Error: Missing label encoder at {ENCODER_PATH}.")
        return

    with open(ENCODER_PATH, "rb") as f:
        mlb = pickle.load(f)
        
    val_dataset = EvaluationDataset(VAL_FILE, tokenizer, mlb)
    loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Safety Check: Verify dimensions align seamlessly
    if model.config.num_labels != len(mlb.classes_):
        print(f"Critical Error: Model labels ({model.config.num_labels}) mismatch Encoder classes ({len(mlb.classes_)}).")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels, all_logits = [], []
    print("Extracting fresh model inferences on CUDA cores...")
    
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy()
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits = model(**inputs).logits
            all_labels.append(labels)
            all_logits.append(logits.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_probs = torch.sigmoid(torch.tensor(np.vstack(all_logits))).numpy()

    print("\nEvaluating Threshold Grid for Metric Optimization:\n")
    best_f1 = -1
    best_thresh = 0.50
    
    # Grid loop searching from 0.10 through 0.95 to locate the true maximum F1 score
    for thresh in np.arange(0.10, 0.98, 0.05):
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        prec = precision_score(all_labels, preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, preds, average="macro", zero_division=0)
        print(f"Threshold: {thresh:.2f} | Macro F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nRecommended Best Macro Threshold: {best_thresh:.2f} (Macro F1: {best_f1:.4f})")
    print(f"\nDetailed Per-Class Classification Report at Threshold {best_thresh:.2f}:\n")
    
    best_preds = (all_probs >= best_thresh).astype(int)
    print(classification_report(all_labels, best_preds, target_names=mlb.classes_, zero_division=0))

if __name__ == "__main__":
    main()