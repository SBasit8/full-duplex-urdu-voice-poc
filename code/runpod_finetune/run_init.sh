#!/bin/bash
# run_init.sh — Download Moshi base model from HF, modify weights for Urdu, save init checkpoint.
# Run on the GPU pod after setup.sh.
# Output: /workspace/init_models/moshiko-urdu-float32/ (~30 GB)
set -e

cd /workspace/moshi-finetune

echo "=== Initializing Moshi for Urdu fine-tune ==="
echo "  - Downloading kyutai/moshiko-pytorch-bf16 from HF (~15 GB)"
echo "  - Re-initializing text embeddings (Urdu SPM has different IDs than Kyutai's English)"
echo "  - Adding Speaker B output heads (full-duplex dual-stream)"
echo "  - Saving as float32 for training stability"
echo ""

python -m tools.init_moshi_for_ft \
    --moshi_lm_repo kyutai/moshiko-pytorch-bf16 \
    --save_dir /workspace/init_models/moshiko-urdu-float32 \
    --model_dtype float32 \
    --init_text_embeddings \
    --extend_modules_for_user_stream

echo ""
echo "=== Init complete ==="
ls -la /workspace/init_models/moshiko-urdu-float32/
echo ""
echo "Next: bash run_finetune.sh"
