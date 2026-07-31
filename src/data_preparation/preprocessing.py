import os
import json
import pickle
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# --- ABSOLUTE PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw/econbiz_stw.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/processed")

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.json")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.json")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.pkl")

# categories from occupying slots in your 100-neuron prediction head.
EXCLUDED_TRAINING_TAGS = {
    "united states", "germany", "theory", "world", 
    "estimation theory", "europe", "eu countries"
}

def main():
    print(f"Running Text Name Preprocessor on: {RAW_DATA_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Cannot find {RAW_DATA_PATH}. Please run your text name conversion script first.")
        return

    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed_texts = []
    processed_labels = []

    for item in raw_data:
        abstract_field = item.get("abstract", "")
        
        # Join arrayed sentence structures seamlessly into unified abstract text paragraphs
        if isinstance(abstract_field, list):
            actual_abstract = " ".join([str(s).strip() for s in abstract_field if s])
        else:
            actual_abstract = str(abstract_field).strip()
            
        raw_subjects = item.get("subject", [])
        
        if not actual_abstract or not raw_subjects:
            continue

        # FIXED: Clean and stringify labels uniformly while stripping background noise tags
        clean_subjects = []
        for s in raw_subjects:
            normalized_tag = str(s).strip()
            # If the tag is one of our blacklisted background noise terms, skip it completely
            if normalized_tag.lower() in EXCLUDED_TRAINING_TAGS:
                continue
            if normalized_tag:
                clean_subjects.append(normalized_tag)

        if clean_subjects:
            processed_texts.append(actual_abstract)
            processed_labels.append(clean_subjects)

    # 1. Identify the Top 100 absolute most frequent subject names (excluding noise)
    all_labels_flat = [label for sublist in processed_labels for label in sublist]
    label_counts = Counter(all_labels_flat)
    
    # Extract the top 100 valid tags
    top_100_names = set(label for label, _ in label_counts.most_common(100))
    
    print(f"\nUnique Subject Name space size discovered in raw file: {len(label_counts)}")
    print("Top 5 ACCURACY-OPTIMIZED Categories detected in dataset:")
    for name, count in label_counts.most_common(5):
        print(f"  - {name}: {count} occurrences")

    final_texts = []
    final_labels = []

    # 2. Filter document labels strictly down to the Top 100 text categories
    for text, labels in zip(processed_texts, processed_labels):
        filtered_item_labels = [l for l in labels if l in top_100_names]
        
        # Only save the abstract record if it contains at least one of our top 100 target categories
        if filtered_item_labels:
            final_texts.append(text)
            final_labels.append(filtered_item_labels)

    print(f"\nFiltered data volume size: {len(final_texts)} records.")

    # 3. Fit the MultiLabelBinarizer strictly on your Top 100 subject strings
    mlb = MultiLabelBinarizer(classes=sorted(list(top_100_names)))
    mlb.fit(final_labels)  

    # Save your text-based configuration metadata matrix
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(mlb, f)
    print(f"Saved text-name configuration metadata matrix to: {ENCODER_PATH}")

    # 4. Train/Val Split (80/20) preserving text arrays natively
    X_train, X_val, y_train, y_val = train_test_split(
        final_texts, final_labels, test_size=0.2, random_state=42
    )

    # 5. Map the raw filtered text string lists directly into the training JSON outputs
    train_dataset = [{"abstract": txt, "subject": labels} for txt, labels in zip(X_train, y_train)]
    val_dataset = [{"abstract": txt, "subject": labels} for txt, labels in zip(X_val, y_val)]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, indent=2, ensure_ascii=False)
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        json.dump(val_dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSUCCESS! Verified Data Assets Created: {len(train_dataset)} Train | {len(val_dataset)} Validation rows.")
    print(f"Target Matrix Vector Width is strictly locked at: {len(mlb.classes_)} unique text headers.")

if __name__ == "__main__":
    main()