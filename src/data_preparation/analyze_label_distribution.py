import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

INPUT_PATH = "../../data/raw/econbiz_stw.json"
PLOT_PATH = "../../data/processed/label_distribution_full.png"

def main():
    print(f"Loading data from {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_labels = []
    for obj in data:
        all_labels.extend(obj["subject"])

    label_counts = Counter(all_labels)
    counts = np.array(list(label_counts.values()))

    print("--- Label Distribution Analysis ---")
    print(f"Total unique labels: {len(label_counts)}")
    thresholds = [1000, 500, 200, 100, 50, 20, 10, 5]
    for t in thresholds:
        print(f"Labels with > {t} samples: {np.sum(counts > t)}")
    
    # Plotting the full distribution
    plt.figure(figsize=(12, 6))
    plt.hist(counts, bins=np.logspace(np.log10(1), np.log10(max(counts)), 50), log=True)
    plt.xscale('log')
    plt.title("Full Label Frequency Distribution (Log-Log Scale)")
    plt.xlabel("Number of Samples per Label")
    plt.ylabel("Number of Labels")
    plt.grid(True, which="both", ls="--")
    plt.savefig(PLOT_PATH)
    print(f"\nSaved full distribution plot to {PLOT_PATH}")

if __name__ == "__main__":
    main() 