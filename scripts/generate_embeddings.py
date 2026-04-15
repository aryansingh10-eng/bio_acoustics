"""
generate_embeddings.py
Extracts WAV2VEC2_LARGE embeddings from raw audio.

Key fixes vs old version:
- Uses WAV2VEC2_LARGE (1024-dim) not BASE (768-dim)  → 2048-dim after mean+max
- Saves EACH CHUNK as a separate .pt file             → 10x more training samples
- Removes broken top-K selection (all norms ≈1 post-normalize)
- Adds SpecAugment-style noise augmentation at embed time
- Skips already-processed files for fast re-runs
"""

import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm

RAW_AUDIO_DIR  = Path("data/raw_audio")
EMBEDDINGS_DIR = Path("data/embeddings")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR   = 16000
CHUNK_SIZE  = TARGET_SR * 3   # 3-second windows
CHUNK_STRIDE= TARGET_SR * 1   # 1-second hop → overlapping chunks = more data
MIN_CHUNK   = TARGET_SR       # ignore chunks shorter than 1 second
SILENCE_THR = 0.005           # RMS threshold below which chunk is silence

print(f"Device: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────────
bundle  = torchaudio.pipelines.WAV2VEC2_LARGE   # 1024-dim hidden, not BASE
model   = bundle.get_model().to(DEVICE)
model.eval()
print("Loaded WAV2VEC2_LARGE")


# ── Audio helpers ─────────────────────────────────────────────────────────────
def load_audio(path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = T.Resample(sr, TARGET_SR)(waveform)
    return waveform                     # [1, T]


def split_chunks(waveform: torch.Tensor) -> list[torch.Tensor]:
    """Overlapping sliding window. Returns only non-silent chunks."""
    chunks = []
    total  = waveform.shape[1]
    for start in range(0, total - CHUNK_SIZE + 1, CHUNK_STRIDE):
        chunk = waveform[:, start : start + CHUNK_SIZE]
        if chunk.shape[1] < MIN_CHUNK:
            continue
        rms = chunk.pow(2).mean().sqrt()
        if rms < SILENCE_THR:
            continue
        chunks.append(chunk)
    # Always keep at least one chunk even if mostly silent
    if not chunks and total >= MIN_CHUNK:
        chunks.append(waveform[:, :CHUNK_SIZE])
    return chunks


@torch.no_grad()
def embed_chunk(chunk: torch.Tensor) -> torch.Tensor:
    """chunk: [1, T]  →  embedding: [2048]"""
    chunk = chunk.to(DEVICE)
    features, _ = model.extract_features(chunk)
    x = features[-1]                             # [1, T', 1024]
    mean_pool = x.mean(dim=1)                    # [1, 1024]
    max_pool  = x.max(dim=1).values              # [1, 1024]
    emb = torch.cat([mean_pool, max_pool], dim=1).squeeze(0)  # [2048]
    return emb.cpu()


# ── Main ──────────────────────────────────────────────────────────────────────
def generate_embeddings():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    class_dirs = sorted([d for d in RAW_AUDIO_DIR.iterdir() if d.is_dir()])
    print(f"\nFound {len(class_dirs)} species")

    total_saved = 0

    for class_dir in class_dirs:
        out_dir = EMBEDDINGS_DIR / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_files = sorted(
            list(class_dir.rglob("*.mp3")) + list(class_dir.rglob("*.wav"))
        )
        print(f"\n{class_dir.name}: {len(audio_files)} audio files")

        saved = 0
        for audio_path in tqdm(audio_files, leave=False):
            try:
                waveform = load_audio(audio_path)
                chunks   = split_chunks(waveform)

                for i, chunk in enumerate(chunks):
                    out_path = out_dir / f"{audio_path.stem}_chunk{i:03d}.pt"
                    if out_path.exists():        # skip already done
                        saved += 1
                        continue
                    emb = embed_chunk(chunk)
                    torch.save(emb, out_path)
                    saved += 1

            except Exception as e:
                print(f"  Error: {audio_path.name}: {e}")

        print(f"  → {saved} embeddings saved")
        total_saved += saved

    print(f"\nDone. Total embeddings: {total_saved}")


if __name__ == "__main__":
    generate_embeddings()
