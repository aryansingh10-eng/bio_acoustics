"""
dataset.py
EmbeddingDataset — loads precomputed WAV2VEC2 embeddings from disk.

Improvements:
- Normalises ONCE (not again in training loop)
- Optional Gaussian noise augmentation for training set
- WeightedRandomSampler weights exposed for the trainer
- Handles corrupt .pt files gracefully (returns zero embedding)
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        embeddings_dir: str,
        min_samples: int = 5,
        augment: bool = False,
        noise_std: float = 0.02,
    ):
        self.augment   = augment
        self.noise_std = noise_std
        self.samples: list = []
        self.label_to_index: dict = {}
        self.index_to_label: dict = {}

        embeddings_dir = str(embeddings_dir)
        if not os.path.exists(embeddings_dir):
            raise FileNotFoundError(
                f"Embeddings directory not found: {embeddings_dir}\n"
                "Run generate_embeddings.py first."
            )

        species_files: dict = {}
        for name in sorted(os.listdir(embeddings_dir)):
            path = os.path.join(embeddings_dir, name)
            if not os.path.isdir(path):
                continue
            pts = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(".pt")
            ]
            if len(pts) >= min_samples:
                species_files[name] = pts
            else:
                print(f"[Dataset] Skipping '{name}' — {len(pts)} samples < {min_samples}")

        if not species_files:
            raise RuntimeError(
                f"No species with >= {min_samples} samples in {embeddings_dir}"
            )

        for idx, name in enumerate(sorted(species_files)):
            self.label_to_index[name] = idx
            self.index_to_label[idx]  = name

        for name, files in species_files.items():
            label = self.label_to_index[name]
            for fp in files:
                self.samples.append((fp, label))

        print(
            f"[Dataset] {len(self.samples)} embeddings, "
            f"{len(self.label_to_index)} species"
            + (" [augmented]" if augment else "")
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # ── Load with corrupt-file protection ────────────────────────────────
        try:
            data = torch.load(path, weights_only=True)
        except Exception:
            # Corrupt file — return zero embedding, training continues cleanly
            return torch.zeros(2048), torch.tensor(label, dtype=torch.long)

        emb = data["embedding"] if isinstance(data, dict) else data
        emb = emb.view(-1).float()

        # Sanitise NaN / Inf
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            emb = torch.zeros_like(emb)

        # Normalise once
        emb = F.normalize(emb, dim=0)

        # Optional augmentation (training only)
        if self.augment:
            emb = emb + torch.randn_like(emb) * self.noise_std
            emb = F.normalize(emb, dim=0)

        return emb, torch.tensor(label, dtype=torch.long)

    # ── Helper for WeightedRandomSampler ─────────────────────────────────────
    def sample_weights(self) -> torch.Tensor:
        """Per-sample weights — inverse class frequency."""
        labels  = [s[1] for s in self.samples]
        counts  = Counter(labels)
        n       = len(self.label_to_index)
        class_w = torch.tensor([1.0 / counts[i] for i in range(n)])
        return class_w[labels]
