import json
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATA_PATH = os.path.join(BASE_DIR, 'data/raw/econbiz_stw.json')
WORDS_PATH = os.path.join(BASE_DIR, 'data/raw/stw-en.tsv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data/filtered/english_stw_filtered.json')

def load_json_data(file_path):
    print(f"Securely loading structural JSON array from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_words(file_path):
    """
    Parses your vocabulary file to build an optimized set of English subject names.
    """
    english_subject_names = set()
    print(f"Extracting reference vocabulary terms from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            parts = line.split("\t")
            # Grab the second column which holds the English text name
            subject_name = parts[1].strip()
            if subject_name:
                english_subject_names.add(subject_name)
                
    print(f"Loaded {len(english_subject_names)} clean reference subject names.")
    return english_subject_names

def check_subjects_in_words(subjects, words_set):
    """
    Evaluates whether the paper contains at least one target concept.
    """
    return any(str(subject).strip() in words_set for subject in subjects)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Missing raw dataset asset at {DATA_PATH}.")
        return
        
    if not os.path.exists(WORDS_PATH):
        print(f"Error: Missing vocabulary file asset at {WORDS_PATH}.")
        return

    # Load resources
    words_set = load_words(WORDS_PATH)
    raw_records = load_json_data(DATA_PATH)

    filtered_data = []
    print("Commencing high-density record intersection matching...")

    for count, record in enumerate(raw_records, 1):
        subjects = record.get('subject', [])
        
        # Subject matching boundary logic check
        if not check_subjects_in_words(subjects, words_set):
            continue
            
        abstract_field = record.get('abstract', '')
        
        # Handle both string values and array-nested string elements safely
        if isinstance(abstract_field, list):
            abstract = " ".join([str(s).strip() for s in abstract_field if s])
        else:
            abstract = str(abstract_field).strip()
            
        if not abstract:
            continue
            
        # Append clean dictionary metadata fields to your training assets
        filtered_data.append({
            'abstract': abstract,
            'subject': subjects
        })
        
        if count % 50000 == 0:
            print(f"Processed {count} validation lines...")

    # Output processing results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
    print(f"SUCCESS! Saved {len(filtered_data)} verified matching rows to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
