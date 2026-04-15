# Run: python scripts/setup_ffmpeg.py
import imageio_ffmpeg, os, subprocess, sys

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)

# Add to current session
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

# Verify
result = subprocess.run([ffmpeg_exe, "-version"], capture_output=True, text=True)
print("ffmpeg found:", ffmpeg_exe)
print(result.stdout[:100])

# Write a .pth file so Python always finds it
import site
pth = os.path.join(site.getsitepackages()[0], "ffmpeg_path.pth")
with open(pth, "w") as f:
    f.write(f"import os; os.environ['PATH'] = r'{ffmpeg_dir}' + os.pathsep + os.environ.get('PATH','')\n")
print(f"Written: {pth}")
print("Done — restart terminal and try again")