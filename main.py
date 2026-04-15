import os
import torch
RAW_AUDIO_PATH="data\raw_audio"
EMBEDDING_PATH="data\embeddings"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

