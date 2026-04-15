"""
fix_embeddings.py
One-time utility to find and delete corrupt .pt embedding files.

Run this BEFORE training if you see:
    OSError: [Errno 22] Invalid argument
    RuntimeError: stack expects each tensor to be equal size

Usage:
    python scripts/fix_embeddings.py
"""

import torch
from pathlib import Path

EMB_DIR = Path("data/embeddings")

print(f"Scanning: {EMB_DIR}")
all_files = list(EMB_DIR.rglob("*.pt"))
print(f"Found {len(all_files)} .pt files\n")

bad   = []
sizes = {}

for pt in all_files:
    try:
        data = torch.load(pt, weights_only=True)
        emb  = data["embedding"] if isinstance(data, dict) else data
        emb  = emb.view(-1)
        size = emb.shape[0]
        sizes[size] = sizes.get(size, 0) + 1
    except Exception as e:
        print(f"  Corrupt: {pt}  ({e})")
        bad.append(pt)

# Report size distribution
print("\nEmbedding size distribution:")
for size, count in sorted(sizes.items()):
    print(f"  {size}-dim : {count} files")

# Flag wrong-size files (keep only 2048-dim)
wrong_size = []
for pt in all_files:
    if pt in bad:
        continue
    try:
        data = torch.load(pt, weights_only=True)
        emb  = data["embedding"] if isinstance(data, dict) else data
        if emb.view(-1).shape[0] != 2048:
            wrong_size.append(pt)
    except Exception:
        pass

print(f"\nCorrupt files  : {len(bad)}")
print(f"Wrong-size files (not 2048-dim): {len(wrong_size)}")

to_delete = bad + wrong_size

if not to_delete:
    print("\nAll files are clean!")
else:
    confirm = input(f"\nDelete {len(to_delete)} bad files? [y/N]: ").strip().lower()
    if confirm == "y":
        for f in to_delete:
            try:
                f.unlink()
            except Exception as e:
                print(f"  Could not delete {f}: {e}")
        print(f"Deleted {len(to_delete)} files.")
    else:
        print("Aborted — no files deleted.")
