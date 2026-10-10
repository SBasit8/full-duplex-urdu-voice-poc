#!/bin/bash
# setup.sh — Run this ONCE on the RunPod GPU instance after SSH-ing in.
set -e

echo "=== Installing WhisperX + dependencies ==="
pip install --upgrade pip
pip install whisperx torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper

echo "=== Verifying GPU ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

echo "=== Verifying WhisperX ==="
python -c "import whisperx; print('WhisperX imported OK')"

echo ""
echo "=== Setup complete ==="
echo "Now run:  python process_all.py --input_dir /path/to/stereo_dialogues_cv --output_dir /path/to/word_timestamps"
