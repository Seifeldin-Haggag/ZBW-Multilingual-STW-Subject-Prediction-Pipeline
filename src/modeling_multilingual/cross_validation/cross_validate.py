import json
import os
import pickle
import numpy as np
from sklearn.model_selection import KFold
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score

# Paths
DATA_PATH = "data/processed/all_data.json"  # Placeholder, update as needed
ENCODER_PATH = "data/processed/label_encoder.pkl"
MODEL_BASE_PATH = "data/processed/cv_models/"
RESULTS_PATH = os.path.join(MODEL_BASE_PATH, "cv_results.json")

# Ensure output directory exists
os.makedirs(MODEL_BASE_PATH, exist_ok=True)

# Load label encoder
with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)

def load_all_data():
    data_dir = "data/processed"
    splits = ["train.json", "val.json", "test.json"]
    all_texts = []
    all_labels = []
    for split in splits:
        with open(os.path.join(data_dir, split), "r") as f:
            data = json.load(f)
            all_texts.extend(data["X"])
            all_labels.extend(data["y"])
    return all_texts, all_labels

class SubjectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
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
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.sigmoid(torch.tensor(pred.predictions)).numpy()
    preds = (preds >= 0.5).astype(int)
    precision = precision_score(labels, preds, average="micro")
    recall = recall_score(labels, preds, average="micro")
    f1 = f1_score(labels, preds, average="micro")
    return {"f1_score": f1, "precision": precision, "recall": recall}

if __name__ == "__main__":
    texts, labels = load_all_data()
    texts = np.array(texts)
    labels = np.array(labels)

    n_splits = 3  # Reduced from 5 to 3 due to limited disk space
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        X_train, X_val = texts[train_idx], texts[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        train_dataset = SubjectDataset(X_train, y_train, tokenizer)
        val_dataset = SubjectDataset(X_val, y_val, tokenizer)

        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

        model = XLMRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(mlb.classes_)
        )

        fold_model_path = os.path.join(MODEL_BASE_PATH, f"fold_{fold+1}")
        training_args = TrainingArguments(
            output_dir=fold_model_path,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            save_steps=500,
            logging_dir=f"{fold_model_path}/logs",
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="f1_score",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train
        trainer.train()

        # Evaluate and get best metrics
        metrics_history = trainer.state.log_history
        best_f1 = 0
        best_metrics = {"f1_score": 0, "precision": 0, "recall": 0}
        for entry in metrics_history:
            if "eval_f1_score" in entry and entry["eval_f1_score"] > best_f1:
                best_f1 = entry["eval_f1_score"]
                best_metrics = {
                    "f1_score": entry["eval_f1_score"],
                    "precision": entry.get("eval_precision", 0),
                    "recall": entry.get("eval_recall", 0)
                }
        fold_metrics.append(best_metrics)
        print(f"Best metrics for fold {fold+1}: {best_metrics}")

        # Save model
        trainer.save_model(fold_model_path)
        tokenizer.save_pretrained(fold_model_path)
        print(f"Model and tokenizer for fold {fold+1} saved to {fold_model_path}")

    # Aggregate metrics
    f1s = [m["f1_score"] for m in fold_metrics]
    precisions = [m["precision"] for m in fold_metrics]
    recalls = [m["recall"] for m in fold_metrics]
    summary = {
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "folds": fold_metrics
    }
    print("\n===== Cross-Validation Summary =====")
    print(json.dumps(summary, indent=2))
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {RESULTS_PATH}") 