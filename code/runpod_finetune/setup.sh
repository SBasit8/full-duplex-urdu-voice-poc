#!/bin/bash
# setup.sh — Install moshi-finetune dependencies on the RunPod GPU pod.
# Run ONCE per pod after SSH-ing in.
set -e

echo "=== [1/5] Upgrading pip ==="
pip install --upgrade pip

echo "=== [2/5] Installing torch (CUDA 12.1) ==="
pip install "torch>=2.4.1" "torchaudio>=2.4.1" --extra-index-url https://download.pytorch.org/whl/cu121

echo "=== [3/5] Installing moshi + tokenizer libs ==="
pip install moshi==0.1.0 sentencepiece huggingface_hub tqdm safetensors

echo "=== [4/5] Installing fine-tune deps (accelerate + deepspeed + transformers) ==="
pip install \
    accelerate==1.1.1 \
    datasets==3.1.0 \
    deepspeed==0.15.4 \
    transformers==4.46.1 \
    wandb soundfile

echo "=== [5/5] Verifying ==="
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.device_count(), 'GPUs')"
python -c "from moshi.models import loaders; print('moshi.loaders OK')"
python -c "import deepspeed; print('deepspeed:', deepspeed.__version__)"
python -c "import accelerate; print('accelerate:', accelerate.__version__)"

echo ""
echo "=== Setup complete ==="
echo "Next:"
echo "  1) huggingface-cli login        # paste HF token"
echo "  2) bash run_init.sh             # download Moshi + init for Urdu (~5-10 min)"
echo "  3) bash run_finetune.sh         # the actual training"
