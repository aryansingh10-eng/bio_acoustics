"""
train_classifier.py
Trains BirdClassifier on precomputed WAV2VEC2_LARGE embeddings.

Speed improvements:
- Mixed precision training (AMP) — ~2x faster on CUDA
- Batch size 256 — fewer iterations per epoch
- Reduced epochs to 60 + patience 7 — faster convergence
"""

import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from dataset import EmbeddingDataset
from models  import BirdClassifier

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = "data/embeddings"
BATCH_SIZE     = 256        # was 64
EPOCHS         = 60         # was 100
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
PATIENCE       = 7          # was 10
MIN_SAMPLES    = 5
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
full_ds = EmbeddingDataset(EMBEDDINGS_DIR, min_samples=MIN_SAMPLES, augment=False)
labels  = [full_ds.samples[i][1] for i in range(len(full_ds))]
num_classes = len(full_ds.label_to_index)

train_idx, val_idx = train_test_split(
    list(range(len(full_ds))),
    test_size=0.2,
    stratify=labels,
    random_state=42,
)

train_ds     = EmbeddingDataset(EMBEDDINGS_DIR, min_samples=MIN_SAMPLES, augment=True)
train_subset = Subset(train_ds, train_idx)
val_subset   = Subset(full_ds,  val_idx)

# WeightedRandomSampler — balanced batches
train_weights = full_ds.sample_weights()[train_idx]
sampler = WeightedRandomSampler(
    train_weights, num_samples=len(train_idx), replacement=True
)

train_loader = DataLoader(
    train_subset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_subset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True
)

# Save label map
with open("label_to_index.json", "w") as f:
    json.dump(full_ds.label_to_index, f, indent=2)
print(f"Saved label_to_index.json ({num_classes} species)")

# ── Model ─────────────────────────────────────────────────────────────────────
sample, _ = full_ds[0]
model = BirdClassifier(input_dim=sample.shape[0], num_classes=num_classes).to(DEVICE)
print(f"Model input dim: {sample.shape[0]}")

# ── Optimizer, scheduler, scaler ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2
)
# Mixed precision scaler — only active on CUDA, no-op on CPU
scaler = GradScaler(enabled=DEVICE.type == "cuda")

# ── Training loop ─────────────────────────────────────────────────────────────
best_val   = 0.0
patience_c = 0
history    = []

for epoch in range(1, EPOCHS + 1):

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    correct  = total = 0
    run_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(enabled=DEVICE.type == "cuda"):
            out  = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        correct  += (out.argmax(1) == y).sum().item()
        total    += y.size(0)
        run_loss += loss.item() * y.size(0)

    train_acc  = 100 * correct / total
    train_loss = run_loss / total
    scheduler.step()

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y  = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

    val_acc = 100 * correct / total
    lr_now  = scheduler.get_last_lr()[0]

    history.append({
        "epoch":      epoch,
        "train_acc":  round(train_acc,  2),
        "val_acc":    round(val_acc,    2),
        "train_loss": round(train_loss, 4),
    })

    print(
        f"Epoch {epoch:3d} | Train {train_acc:5.1f}% | Val {val_acc:5.1f}% "
        f"| Loss {train_loss:.4f} | LR {lr_now:.2e}"
    )

    if val_acc > best_val:
        best_val   = val_acc
        patience_c = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✓ Saved (best val: {best_val:.1f}%)")
    else:
        patience_c += 1
        if patience_c >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {PATIENCE} epochs)"
            )
            break

# ── Save history ──────────────────────────────────────────────────────────────
with open("history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nBest val accuracy: {best_val:.1f}%")
print("Saved: best_model.pth, history.json")
