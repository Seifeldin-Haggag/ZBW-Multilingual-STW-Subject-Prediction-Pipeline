import os
import json
import torch
import random
import pickle
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)

# --- ROBUST ABSOLUTE PATH NAVIGATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATA_DIR_PROCESSED = os.path.join(PROJECT_ROOT, "data/processed")
MODEL_SAVE_PATH = os.path.join(DATA_DIR_PROCESSED, "xlm_roberta_trained_model")
TRAIN_FILE = os.path.join(DATA_DIR_PROCESSED, "train.json")
VAL_FILE = os.path.join(DATA_DIR_PROCESSED, "val.json")
ENCODER_PATH = os.path.join(DATA_DIR_PROCESSED, "label_encoder.pkl")

# --- HYPERPARAMETERS ---
BATCH_SIZE = 8           
EPOCHS = 5               
LEARNING_RATE = 2e-5    
MAX_LENGTH = 256
TRAIN_SAMPLE_SIZE = None 

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def load_data_list(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["abstract"] for item in data]
    labels = [item["subject"] for item in data]
    return texts, labels

class SubjectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, mlb, max_length=256):
        self.texts = texts
        self.labels = mlb.transform(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.sigmoid(torch.tensor(pred.predictions)).numpy()
    preds = (preds >= 0.50).astype(int)  
    
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

# =====================================================================
# ACCURACY OPTIMIZED TRANSITION ENGINE: MULTI-LABEL FOCAL LOSS TRAINER
# =====================================================================
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Overrides loss updates with dynamically tracking Focal metrics."""
        labels = inputs.get("labels")
        
        # Safe forward pass copy generation
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        probs = torch.sigmoid(logits)
        
        # Native BCE logits compilation
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction='none'
        )
        
        # Focal Modulation: scales down error on easy background choices
        p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
        focal_weight = (1.0 - p_t) ** 2.0  # Gamma locked at 2.0
        
        # Balanced Alpha adjustment 
        loss = 0.25 * focal_weight * bce_loss
        loss = loss.mean()
        
        return (loss, outputs) if return_outputs else loss

def train_model():
    device_str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Target Device Hardware Detected: {device_str.upper()}")
    
    train_texts, train_labels = load_data_list(TRAIN_FILE)
    val_texts, val_labels = load_data_list(VAL_FILE)
    
    if TRAIN_SAMPLE_SIZE is not None and TRAIN_SAMPLE_SIZE < len(train_texts):
        indices = random.sample(range(len(train_texts)), TRAIN_SAMPLE_SIZE)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]

    if not os.path.exists(ENCODER_PATH):
        print(f"Error: Cannot find your encoder at {ENCODER_PATH}. Please run your preprocessing script first!")
        return

    with open(ENCODER_PATH, "rb") as f:
        mlb = pickle.load(f)

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    
    train_dataset = SubjectDataset(train_texts, train_labels, tokenizer, mlb, max_length=MAX_LENGTH)
    val_dataset = SubjectDataset(val_texts, val_labels, tokenizer, mlb, max_length=MAX_LENGTH)

    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification"
    )
    
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,      # Enabled rollback capability
        eval_strategy="epoch",            
        save_strategy="epoch",            
        fp16=True if device_str == "cuda" else False, 
        metric_for_best_model="f1_score",
        greater_is_better=True,
        dataloader_pin_memory=False if device_str == "mps" else True, 
        report_to="none"
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    print("Commencing Fine-Tuning Execution Pipeline...")
    trainer.train()
    
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f" Model weights saved securely to: {MODEL_SAVE_PATH}")

    # Generate sample evaluation diagnostics
    print("\nDisplaying Sample Pipeline Diagnostic Predictions:")
    
    device = torch.device(device_str)
    model.to(device)
    model.eval()
    
    val_indices = list(range(len(val_dataset)))
    random.shuffle(val_indices)
    sample_targets = val_indices[:5]
    
    for count, idx in enumerate(sample_targets, 1):
        item = val_dataset[idx]
        
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].to(device).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
        # Evaluation Diagnostic print gates matched to 0.50 standard parameters
        pred_indices = (probs >= 0.50).nonzero()[0]
        pred_labels = [mlb.classes_[i] for i in pred_indices]
        top5_indices = probs.argsort()[-5:][::-1]
        top5_labels = [(mlb.classes_[i], float(probs[i])) for i in top5_indices]
        
        true_labels = val_labels[idx]
        
        print(f"\nSample {count}:")
        print("Abstract Text Preview:", val_texts[idx][:200], "...")
        print("True Document Target Labels:", true_labels)
        print("Predicted Labels (0.50 Baseline):", pred_labels)
        print("Top 5 Confidence Probs:", top5_labels)

if __name__ == "__main__":
    train_model()