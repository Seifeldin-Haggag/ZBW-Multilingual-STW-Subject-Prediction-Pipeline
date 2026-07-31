import os
import json
import torch
import pickle
import random
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# --- ABSOLUTE PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")
MODEL_PATH = os.path.join(DATA_DIR, "xlm_roberta_trained_model")
TRAIN_FILE = os.path.join(DATA_DIR, "train.json")

# Confirmed: Tracking your main active encoder file
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")

# Using your optimized threshold boundary found during validation calibrating
THRESHOLD = 0.45

def main():
    print("🔮 Initializing Batch Random Text-Name Inference Engine...")
    
    # Detect Apple Silicon MPS backend or standard Linux CUDA GPU configurations
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"💻 Running calculations using device architecture: {device.type.upper()}")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH) or not os.path.exists(TRAIN_FILE):
        print(f"❌ Error: Required processed training assets missing from paths.")
        return

    # Load Model assets
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    with open(ENCODER_PATH, "rb") as f:
        mlb = pickle.load(f)
        
    print("✅ Text category maps and neural layers successfully linked into memory!")
    print(f"📌 Binarizer Class Count Locked At: {len(mlb.classes_)} unique headers.")

    # Load and randomly sample 20 abstracts from training data
    with open(TRAIN_FILE, "r", encoding='utf-8') as f:
        data = json.load(f)
    samples = random.sample(data, min(20, len(data)))

    print(f"\n🚀 Generation sequence started for {len(samples)} random test records...\n")

    for i, sample in enumerate(samples, 1):
        raw_abstract = sample.get("abstract", [])
        
        # Unify abstract sentence lists into a solid text block
        if isinstance(raw_abstract, list):
            abstract_text = " ".join(raw_abstract)
        else:
            abstract_text = str(raw_abstract)
            
        if not abstract_text.strip():
            continue

        # Tokenization & device routing
        inputs = tokenizer(
            abstract_text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        # Extract predictions based on threshold criteria
        prediction_mask = (probabilities >= THRESHOLD).astype(int)
        
        # inverse_transform maps directly to whatever data type is inside your label_encoder.pkl
        predicted_names = mlb.inverse_transform(np.array([prediction_mask]))[0]

        # Display clean matching outputs
        print(f"================== SAMPLE {i} ==================")
        print(f"Abstract Preview: {abstract_text[:180]}...")
        print(f"True Subject Tags: {sample.get('subject', [])}")
        
        if not predicted_names:
            print("Predicted Subjects: 📭 No classes passed the threshold activation.")
        else:
            print(f"Predicted Subjects: {list(predicted_names)}")
        print("==============================================\n")

if __name__ == "__main__":
    main()