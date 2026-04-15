"""
fix_birdnet_direct.py — hardcoded paths version
Run: python scripts/fix_birdnet_direct.py
"""

import os
import re

BASE = r"C:\Users\Acer\OneDrive - Manipal Academy of Higher Education\Desktop\Bioacoustics\bioenv\lib\site-packages\birdnetlib"

UTILS_PATH = os.path.join(BASE, "utils.py")
MAIN_PATH  = os.path.join(BASE, "main.py")

print(f"utils.py → {UTILS_PATH}")
print(f"exists   → {os.path.exists(UTILS_PATH)}")

# ── Patch utils.py ────────────────────────────────────────────────────────────
with open(UTILS_PATH, "r", encoding="utf-8") as f:
    utils = f.read()

# Remove any previous patch
if "# LIBROSA_PATCH_START" in utils:
    utils = re.sub(r"# LIBROSA_PATCH_START.*?# LIBROSA_PATCH_END\n", "", utils, flags=re.DOTALL)
    print("Removed old patch from utils.py")

STUB = """\
# LIBROSA_PATCH_START
import sys as _sys, types as _types
if 'librosa' not in _sys.modules:
    import soundfile as _sf, resampy as _resampy
    def _librosa_load(path, sr=None, mono=True, res_type='kaiser_best', **kw):
        data, orig_sr = _sf.read(str(path), always_2d=True, dtype='float32')
        data = data.mean(axis=1) if mono else data.squeeze()
        target = int(sr) if sr else orig_sr
        if target != orig_sr:
            data = _resampy.resample(data, orig_sr, target)
        return data.astype('float32'), target
    _lib = _types.ModuleType('librosa')
    _lib.load = _librosa_load
    _lib.core = _types.ModuleType('librosa.core')
    _lib.core.load = _librosa_load
    _sys.modules['librosa'] = _lib
    _sys.modules['librosa.core'] = _lib.core
# LIBROSA_PATCH_END
"""

with open(UTILS_PATH, "w", encoding="utf-8") as f:
    f.write(STUB + utils)
print("✓ Patched utils.py")

# ── Clean main.py — remove old broken patch ───────────────────────────────────
with open(MAIN_PATH, "r", encoding="utf-8") as f:
    main = f.read()

if "_patched_librosa_load" in main or (
    "import soundfile as _sf" in main and "_patched_librosa_load" in main
):
    for marker in ("import librosa", "import numpy", "import warnings", "import os"):
        idx = main.find("\n" + marker)
        if idx != -1:
            main = main[idx + 1:]
            print(f"✓ Stripped old patch from main.py")
            break
    with open(MAIN_PATH, "w", encoding="utf-8") as f:
        f.write(main)
else:
    print("  main.py: no old patch to remove")

print("\n✓ All done! Now run:  streamlit run scripts/bird_app.py")
