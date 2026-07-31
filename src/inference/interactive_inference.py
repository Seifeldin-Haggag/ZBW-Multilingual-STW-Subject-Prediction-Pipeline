import os
import json
import torch
import pickle
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# --- ABSOLUTE PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")
MODEL_PATH = os.path.join(DATA_DIR, "xlm_roberta_trained_model")
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
LOCAL_STW_MAP_PATH = os.path.join(PROJECT_ROOT, "data/raw/stw-en.tsv")

# Locked at your mathematically optimized threshold sweet-spot
OPTIMAL_THRESHOLD = 0.45

def clean_string_key(key_val):
    """Normalizes classification slugs for reliable dictionary mapping."""
    if key_val is None:
        return ""
    val_str = str(key_val).replace("<", "").replace(">", "").replace("\r", "").replace("\n", "").strip()
    if "/" in val_str:
        val_str = val_str.split("/")[-1].strip()
    return val_str.lower()

def load_local_stw_labels(map_path):
    """Loads stw-en.tsv mappings safely handling both raw strings and URIs."""
    id_to_name = {}
    if not os.path.exists(map_path):
        print(f"Local mapping file missing at: {map_path}\nProceeding with native token format translation.")
        return id_to_name

    with open(map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            parts = line.split("\t")
            raw_uri, label_name = parts[0].strip(), parts[1].strip()
            
            clean_id = clean_string_key(raw_uri)
            if clean_id:
                id_to_name[clean_id] = label_name
    return id_to_name

def main():
    print("Initializing Interactive Multilingual Automated Indexing Shell...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Computing Hardware Context Activated: {device}")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print(f"Deploy Error: Model assets or pickled encoder targets missing at: {DATA_DIR}")
        return

    # Load resources into VRAM/System Memory
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    with open(ENCODER_PATH, "rb") as f:
        mlb = pickle.load(f)
        
    stw_lookup = load_local_stw_labels(LOCAL_STW_MAP_PATH)
    print("Inference engines and concept maps loaded successfully!\n")
    print("=====================================================================")
    print("Type or paste your text below to get subject predictions.")
    print("Type 'exit' or 'quit' anytime to turn off the loop console.")
    print("=====================================================================\n")

    while True:
        try:
            user_input = input("Enter Abstract Text: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("\nShutting down interactive deployment interface. Goodbye! 👋")
                break
                
            if not user_input:
                print("System Warning: Input text block cannot be completely blank.\n")
                continue

            # Tokenize text seamlessly
            inputs = tokenizer(
                user_input, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=256
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]

            # Primary Threshold Evaluation Gateway
            predicted_indices = np.where(probabilities >= OPTIMAL_THRESHOLD)[0]

            # If the user enters a tiny phrase and nothing clears 0.7, return the absolute highest guess
            is_fallback = False
            if len(predicted_indices) == 0:
                highest_idx = np.argmax(probabilities)
                if probabilities[highest_idx] > 0.02:  # Safe low-bound noise cutoff
                    predicted_indices = np.array([highest_idx])
                    is_fallback = True

            predicted_raw_classes = mlb.classes_[predicted_indices]

            print("\n --- TARGET AUTO-INDEXING PREDICTIONS ---")
            if len(predicted_raw_classes) == 0:
                print("Muted: Text context structure generated no confidence signals.")
            else:
                if is_fallback:
                    print(f" [Low-Text Fallback Mode: Showing single highest confidence index (Prob: {probabilities[predicted_indices[0]]:.4f})]")
                
                for raw_class in predicted_raw_classes:
                    lookup_key = clean_string_key(raw_class)
                    name_str = stw_lookup.get(lookup_key)
                    
                    # FALLBACK AUTO-FORMATTER: If the literal key isn't in your TSV file, 
                    # cleanly capitalize and space out the text token instead of throwing 'Unmapped Class'
                    if not name_str:
                        name_str = lookup_key.replace("_", " ").replace("-", " ").title()
                        
                        # Handle targeted country/entity abbreviation formatting overrides
                        if name_str.lower() == "eu countries":
                            name_str = "EU Countries"
                        elif name_str.lower() == "united states":
                            name_str = "United States"
                            
                    print(f"  {name_str}")
            print("-------------------------------------------\n")

        except KeyboardInterrupt:
            print("\nExiting shell loop interface safely.")
            break
        except Exception as e:
            print(f"Runtime Processing Error: {str(e)}\n")

if __name__ == "__main__":
    main()