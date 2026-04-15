import os
import logging
import torch
import torchaudio
import torchaudio.transforms as T

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants (single source of truth) ────────────────────────────────────────
TARGET_SAMPLE_RATE = 16_000
TARGET_DURATION    = 5                                   # seconds
TARGET_NUM_SAMPLES = TARGET_SAMPLE_RATE * TARGET_DURATION
N_MELS             = 64


def preprocess_audio(file_path, device: torch.device) -> torch.Tensor:
    """
    Load an MP3/WAV file and return a log-mel spectrogram tensor.

    Returns
    -------
    torch.Tensor  shape [1, 1, N_MELS, time_frames]
                  ready to be fed directly into AudioEncoder.
    """

    # ── 1. Load ───────────────────────────────────────────────────────────
    waveform, sample_rate = torchaudio.load(str(file_path))
    logger.debug("Loaded %s  shape=%s  sr=%d", file_path, waveform.shape, sample_rate)

    # ── 2. Stereo → mono ─────────────────────────────────────────────────
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # ── 3. Resample to 16 kHz ────────────────────────────────────────────
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform  = resampler(waveform)

    # ── 4. Fix duration to exactly 5 s ───────────────────────────────────
    n = waveform.shape[1]
    if n > TARGET_NUM_SAMPLES:
        waveform = waveform[:, :TARGET_NUM_SAMPLES]
    elif n < TARGET_NUM_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_NUM_SAMPLES - n))

    # ── 5. Peak-normalise amplitude ──────────────────────────────────────
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / (peak + 1e-9)

    # ── 6. Move to device ────────────────────────────────────────────────
    waveform = waveform.to(device)

    # ── 7. Mel spectrogram ───────────────────────────────────────────────
    mel_transform = T.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS,
        f_min=50,       # bird calls rarely go below 50 Hz
        f_max=8000,     # and rarely above 8 kHz
    ).to(device)

    mel_spec = mel_transform(waveform)                   # [1, N_MELS, time]

    # ── 8. Log-mel (numerical stability) ─────────────────────────────────
    mel_spec = torch.log(mel_spec + 1e-9)

    # ── 9. Add batch dimension → [1, 1, N_MELS, time] ───────────────────
    mel_spec = mel_spec.unsqueeze(0)

    return mel_spec


# ── Quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    RAW_AUDIO_PATH = "data/raw_audio"

    species_list = os.listdir(RAW_AUDIO_PATH)
    first_species_path = os.path.join(RAW_AUDIO_PATH, species_list[0])

    sample_file = next(
        os.path.join(first_species_path, f)
        for f in os.listdir(first_species_path)
        if f.lower().endswith(".mp3")
    )

    logger.info("Testing on: %s", sample_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel = preprocess_audio(sample_file, device)
    logger.info("Output shape: %s", mel.shape)
