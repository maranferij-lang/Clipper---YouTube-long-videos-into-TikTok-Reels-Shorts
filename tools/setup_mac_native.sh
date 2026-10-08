#!/bin/bash
# setup_mac_native.sh: one-time setup on macOS. Run in Terminal: bash tools/setup_mac_native.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
brew list node   >/dev/null 2>&1 || brew install node
brew list python@3.11 >/dev/null 2>&1 || brew install python@3.11
[ -d "$HOME/clipper_venv" ] || python3.11 -m venv "$HOME/clipper_venv"
source "$HOME/clipper_venv/bin/activate"
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy scipy opencv-python pillow python_speech_features pandas tqdm yt-dlp
[ -d "$HOME/Light-ASD" ] || git clone -q https://github.com/Junhua-Liao/Light-ASD "$HOME/Light-ASD"
python3 "$HERE/patch_lightasd_cpu.py" "$HOME/Light-ASD/ASD.py"
mkdir -p "$HERE/weight"; cp -n "$HOME/Light-ASD/weight/"*.model "$HERE/weight/" 2>/dev/null || true
mkdir -p "$HOME/.config/yt-dlp"; grep -q js-runtimes "$HOME/.config/yt-dlp/config" 2>/dev/null || echo "--js-runtimes node" >> "$HOME/.config/yt-dlp/config"
python3 -c "import torch, cv2, scipy, python_speech_features; print('ok: torch', torch.__version__, 'cv2', cv2.__version__)"
echo "Done. Still needed in tools/: yunet.onnx (YuNet face detector) and Mv.ttf (a bold font, e.g. Montserrat). See README."
echo "Run a batch: bash $HERE/run_loop.sh <jobs.json>"
