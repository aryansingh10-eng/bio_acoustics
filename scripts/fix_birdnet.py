"""
fix_birdnet.py  —  Permanently patches birdnetlib/utils.py to use
soundfile + resampy instead of librosa (which requires ffmpeg for MP3).

Run ONCE from your venv:
    python fix_birdnet.py

Then restart Streamlit normally.
"""

import sys, re
from pathlib import Path

# ── Locate utils.py ──────────────────────────────────────────────────────────
# Try automatic detection first
try:
    import birdnetlib
    utils_path = Path(birdnetlib.__file__).parent / "utils.py"
except ImportError:
    # Fall back to hard-coded path
    utils_path = Path(
        r"C:\Users\Acer\OneDrive - Manipal Academy of Higher Education"
        r"\Desktop\Bioacoustics\bioenv\lib\site-packages\birdnetlib\utils.py"
    )

if not utils_path.exists():
    sys.exit(f"ERROR: Could not find {utils_path}")

print(f"Found: {utils_path}")

content = utils_path.read_text(encoding="utf-8")

# ── Guard: already patched? ──────────────────────────────────────────────────
if "PATCHED_SF" in content:
    print("Already patched — nothing to do.")
    sys.exit(0)

# ── Step 1: Remove 'import librosa' ─────────────────────────────────────────
if "import librosa" in content:
    content = content.replace("import librosa\n", "# import librosa  [PATCHED_SF]\n")
    print("Removed 'import librosa'")
else:
    print("WARNING: 'import librosa' not found — may already be removed")

# ── Step 2: Replace librosa.load(...) call ───────────────────────────────────
# birdnetlib calls it like:
#   audio_chunk, _ = librosa.load(
#       file_path, sr=..., offset=..., duration=..., res_type=...
#   )
#
# We replace the ENTIRE assignment with a soundfile+resampy equivalent.
# Use regex so we catch any whitespace / line-continuation variations.

pattern = re.compile(
    r'audio_chunk\s*,\s*_\s*=\s*librosa\.load\s*\([^)]*\)',
    re.DOTALL
)

replacement = """\
# [PATCHED_SF] replaced librosa.load with soundfile + resampy
            import soundfile as _sf_mod, resampy as _rsp_mod, numpy as _np_mod
            _sf_info   = _sf_mod.info(str(file_path))
            _orig_sr   = _sf_info.samplerate
            _start_frm = int(offset * _orig_sr) if offset else 0
            _stop_frm  = (int(_start_frm + duration * _orig_sr)
                          if duration else _sf_info.frames)
            _stop_frm  = min(_stop_frm, _sf_info.frames)
            _raw, _     = _sf_mod.read(
                str(file_path), always_2d=True, dtype="float32",
                start=_start_frm, stop=_stop_frm
            )
            _raw = _raw.mean(axis=1)  # stereo → mono
            audio_chunk = (
                _rsp_mod.resample(_raw, _orig_sr, sr, filter="kaiser_best")
                if _orig_sr != sr else _raw
            )
            _ = sr"""

match = pattern.search(content)
if match:
    print(f"Found librosa.load call:\n  {match.group()[:120].strip()}...")
    content = content[:match.start()] + replacement + content[match.end():]
    print("Replaced with soundfile + resampy")
else:
    # Try a simpler literal match as fallback
    old = "audio_chunk, _ = librosa.load("
    if old in content:
        # Find end of this call (closing paren)
        start = content.index(old)
        depth = 0
        end   = start
        for i, ch in enumerate(content[start:], start):
            if ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        print(f"Found via literal match: {content[start:end][:120].strip()}...")
        content = content[:start] + replacement + content[end:]
        print("Replaced with soundfile + resampy")
    else:
        print("\nERROR: Could not find librosa.load call.")
        print("Printing lines containing 'librosa' for manual inspection:")
        for i, line in enumerate(content.splitlines(), 1):
            if "librosa" in line:
                print(f"  Line {i}: {line}")
        sys.exit(1)

# ── Write back ───────────────────────────────────────────────────────────────
utils_path.write_text(content, encoding="utf-8")
print(f"\nDone! Saved {utils_path}")
print("Now run:  python -m streamlit run scripts/bird_app.py")
