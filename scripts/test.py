"""
test.py
Evaluates a trained BirdClassifier on a held-out test set.

Outputs:
  - Overall top-1 and top-3 accuracy
  - Per-class precision, recall, F1
  - Confusion matrix saved to confusion_matrix.png
  - Full results saved to test_results.json
"""

import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from dataset import EmbeddingDataset
from models  import BirdClassifier

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = "data/embeddings"
MODEL_PATH     = "best_model.pth"
LABEL_PATH     = "label_to_index.json"
BATCH_SIZE     = 64
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # ── Load labels ───────────────────────────────────────────────────────────
    with open(LABEL_PATH) as f:
        label_to_index = json.load(f)
    index_to_label = {v: k for k, v in label_to_index.items()}
    num_classes = len(label_to_index)

    # ── Load dataset (no augment) ─────────────────────────────────────────────
    ds = EmbeddingDataset(EMBEDDINGS_DIR, min_samples=5, augment=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Load model ────────────────────────────────────────────────────────────
    sample, _ = ds[0]
    model = BirdClassifier(input_dim=sample.shape[0], num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds   = []
    all_labels  = []
    top3_correct = 0
    total        = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs  = F.softmax(logits, dim=1)

            top1 = probs.argmax(1)
            all_preds.extend(top1.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

            top3 = probs.topk(min(3, num_classes), dim=1).indices
            for i, label in enumerate(y):
                if label in top3[i]:
                    top3_correct += 1
            total += y.size(0)

    top1_acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / total
    top3_acc = 100 * top3_correct / total

    print(f"\nTop-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")

    # ── Per-class report ──────────────────────────────────────────────────────
    class_names = [index_to_label[i] for i in range(num_classes)]
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    print("\nPer-class F1:")
    for name in class_names:
        r = report[name]
        print(f"  {name:40s}  P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1-score']:.2f}  n={int(r['support'])}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(max(8, num_classes), max(6, num_classes - 2)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nSaved: confusion_matrix.png")

    # ── Save JSON results ─────────────────────────────────────────────────────
    results = {
        "top1_acc": round(top1_acc, 2),
        "top3_acc": round(top3_acc, 2),
        "num_samples": total,
        "num_classes": num_classes,
        "per_class": {
            name: {k: round(v, 3) for k, v in report[name].items()}
            for name in class_names
        },
    }
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: test_results.json")


if __name__ == "__main__":
    main()
