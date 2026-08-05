import os
import json
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# --- ABSOLUTE PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/filtered/english_stw_filtered.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/processed")

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.json")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.json")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.pkl")

def main():
    print(f"Running Full-Vocabulary Preprocessor on: {RAW_DATA_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Cannot find {RAW_DATA_PATH}. Please run your filter script first.")
        return

    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed_texts = []
    processed_labels = []

    for item in raw_data:
        abstract_field = item.get("abstract", "")
        
        if isinstance(abstract_field, list):
            actual_abstract = " ".join([str(s).strip() for s in abstract_field if s])
        else:
            actual_abstract = str(abstract_field).strip()
            
        raw_subjects = item.get("subject", [])
        
        if not actual_abstract or not raw_subjects:
            continue

        # KEEP ALL SUBJECTS (NO EXCLUSION BLACKLIST)
        clean_subjects = [str(s).strip() for s in raw_subjects if str(s).strip()]

        if clean_subjects:
            processed_texts.append(actual_abstract)
            processed_labels.append(clean_subjects)

    all_labels_flat = [label for sublist in processed_labels for label in sublist]
    label_counts = Counter(all_labels_flat)
    
    print(f"\nTotal Unique Subject Space Discovered: {len(label_counts)} tags")

    # =========================================================================
    # INCLUDE ALL CLASSES (Minimum frequency threshold = 1 to keep everything)
    # =========================================================================
    MIN_LABEL_FREQUENCY = 1  
    all_valid_names = set(label for label, count in label_counts.items() if count >= MIN_LABEL_FREQUENCY)
    
    print(f"Unlocked ALL categories: {len(all_valid_names)} total target classes")

    final_texts = []
    final_labels = []

    for text, labels in zip(processed_texts, processed_labels):
        filtered_item_labels = [l for l in labels if l in all_valid_names]
        
        if filtered_item_labels:
            final_texts.append(text)
            final_labels.append(filtered_item_labels)

    print(f"Filtered dataset size: {len(final_texts)} records.")

    # Fit MultiLabelBinarizer on the COMPLETE vocabulary
    mlb = MultiLabelBinarizer(classes=sorted(list(all_valid_names)))
    mlb.fit(final_labels)  

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(mlb, f)
    print(f"Saved label encoder metadata matrix to: {ENCODER_PATH}")

    # Train/Val Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        final_texts, final_labels, test_size=0.2, random_state=42
    )

    train_dataset = [{"abstract": txt, "subject": labels} for txt, labels in zip(X_train, y_train)]
    val_dataset = [{"abstract": txt, "subject": labels} for txt, labels in zip(X_val, y_val)]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, indent=2, ensure_ascii=False)
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        json.dump(val_dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSUCCESS! Created {len(train_dataset)} Train | {len(val_dataset)} Validation rows.")
    print(f"Target Classification Head Width: {len(mlb.classes_)} unique classes.")

if __name__ == "__main__":
    main()
