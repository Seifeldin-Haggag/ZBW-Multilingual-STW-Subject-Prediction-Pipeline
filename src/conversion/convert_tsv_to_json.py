import os
import json
import random

# Seed the random number generator for reproducible downsampling results
random.seed(42)

# --- ABSOLUTE PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# Input/Output Data Paths
TSV_PATH = os.path.join(PROJECT_ROOT, "data/raw/econbiz_stw.tsv")
OUTPUT_JSON_PATH = os.path.join(PROJECT_ROOT, "data/raw/econbiz_stw.json")

# Vocabulary Mapping Path (The English mapping file)
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data/raw/stw-en.tsv")

# CONFIGURATION: Overrepresented generic tags hurting model accuracy
DOMINANT_NOISE_TAGS = {"united states", "germany", "theory", "world", "estimation theory"}

def build_translation_dict(vocab_path):
    id_to_name = {}
    if not os.path.exists(vocab_path):
        print(f" Warning: Vocabulary translation file missing at: {vocab_path}")
        print("Fallback mode active: Output will keep numeric IDs if vocabulary is absent.")
        return id_to_name

    print(f"Clean Step 1: Building translation dictionary from: {vocab_path}")
    with open(vocab_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or "\t" not in line:
                continue
                
            parts = line.split("\t")
            uri_part = parts[0].strip().lower() # Casing normalized to lowercase
            name_part = parts[1].strip()
            
            # Isolate the trailing alphanumeric index from the URI string template
            if "descriptor/" in uri_part:
                stw_id = uri_part.split("descriptor/")[-1].replace(">", "").strip()
                id_to_name[stw_id] = name_part
            elif "thsys/" in uri_part:
                stw_id = uri_part.split("thsys/")[-1].replace(">", "").strip()
                id_to_name[stw_id] = name_part
                
    print(f"Map creation finalized! Loaded {len(id_to_name)} ID-to-Name relationships.")
    return id_to_name

def main():
    print(f"Initializing Direct Tab Slicer from: {TSV_PATH}")
    if not os.path.exists(TSV_PATH):
        print(f"Error: Missing stw-econbiz.tsv in data/raw/")
        return

    # 1. Initialize our translation map helper
    id_to_name = build_translation_dict(VOCAB_PATH)
    recovered_records = []
    unmapped_codes_tracker = set()
    
    skipped_noise_count = 0

    # 2. Parse the primary big data TSV file
    with open(TSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            clean_line = line.replace("\n", "").replace("\r", "")
            parts = clean_line.split("\t")
            
            if len(parts) < 2:
                continue
                
            text_block = parts[0].strip()
            raw_labels = parts[1:]  
            
            clean_tags = []
            for tag in raw_labels:
                tag = tag.strip().lower()
                if not tag:
                    continue
                    
                # Clean off the official URI wrappers safely to isolate the pure key
                if "/" in tag:
                    tag = tag.split("/")[-1].replace(">", "").replace("<", "").strip()
                
                if tag:
                    if tag in id_to_name:
                        clean_tags.append(id_to_name[tag])
                    else:
                        clean_tags.append(tag) # Fallback: keep ID if not in stw-en vocabulary
                        unmapped_codes_tracker.add(tag)
                    
            if text_block and clean_tags:
                is_only_noise = all(t.lower().strip() in DOMINANT_NOISE_TAGS for t in clean_tags)
                
                if is_only_noise:
                    # Drop 80% of rows that don't add specific subject value
                    if random.random() > 0.20:
                        skipped_noise_count += 1
                        continue  # Bypass and filter out of the dataset
                
                # If it passed the filter or has unique tags, save it
                recovered_records.append({
                    "abstract": text_block,
                    "subject": clean_tags
                })

            # Safe capping threshold for target 167,000 records pool execution
            if len(recovered_records) >= 167000:
                print(f"Reached optimal target validation size ceiling of {len(recovered_records)} records.")
                break

    print(f"\nExtraction sequence finalized successfully!")
    print(f"Total parsed records written to asset pool array: {len(recovered_records)}")
    print(f"Filtered Out Rows (Useless Generic Noise Bias Drops): {skipped_noise_count}")
    
    if unmapped_codes_tracker:
        print(f"Note: {len(unmapped_codes_tracker)} codes fell back to numeric IDs due to missing map elements.")

    # 3. Output structural JSON file
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(recovered_records, f, indent=2, ensure_ascii=False)
        
    print(f"Balanced Big Data JSON written cleanly to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()