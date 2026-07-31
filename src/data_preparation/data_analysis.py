import json
import os

DATA_PATH = "../../data/raw/econbiz_stw.json"

def analyze_data(file_path):
    num_objects = 0
    all_subjects = []

    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    num_objects = len(data)

    # Collect all subjects
    for item in data:
        subjects = item.get("subject", [])
        all_subjects.extend(subjects)

    unique_subjects = set(all_subjects)

    print(f"\nTotal Objects (Abstracts): {num_objects}")
    print(f"Total Unique Subjects: {len(unique_subjects)}")
    print(f"Sample Subjects: {list(unique_subjects)[:10]}")

    # Display first 2 objects with formatted JSON
    print("\nSample Data (First 2 Objects):\n")
    for obj in data[:2]:
        print(json.dumps(obj, indent=2))
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    analyze_data(DATA_PATH)
