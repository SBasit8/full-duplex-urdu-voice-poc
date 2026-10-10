#!/bin/bash
# setup.sh — Run ONCE on the RunPod GPU instance after SSH-ing in.
set -e

echo "=== Installing Mimi tokenizer + moshi-finetune deps ==="
pip install --upgrade pip
pip install "torch>=2.4.1" torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install moshi==0.1.0 huggingface_hub tqdm numpy soundfile

echo "=== Verifying GPU ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

echo "=== Verifying Mimi loader ==="
python -c "from moshi.models import loaders; print('moshi.loaders OK')"

echo ""
echo "=== Setup complete ==="
echo "Next: python tokenize_audio.py --audio_dir /workspace/stereo_dialogues_flat --output_dir /workspace/tokenized_audio --resume"
