import torchaudio
from pathlib import Path

ROOT = Path("data/raw_audio")

bad_files = []

for audio_file in ROOT.rglob("*.mp3"):
    try:
        torchaudio.load(audio_file)
    except Exception:
        print(f"❌ Removing corrupted file: {audio_file}")
        bad_files.append(audio_file)

# delete bad files
for f in bad_files:
    try:
        f.unlink()
    except:
        pass

print(f"\nRemoved {len(bad_files)} corrupted files")