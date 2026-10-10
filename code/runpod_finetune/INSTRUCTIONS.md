# Moshi Fine-tuning on RunPod — End-to-End Guide

## Overview
This bundle does **install + init + fine-tune** all on the same RunPod pod, avoiding the 30 GB upload of the init model. You upload only ~30 MB total.

## Pre-flight checklist on LOCAL machine

Verify these exist before starting:
```bash
ls -lh "/home/ubuntu/vs_code/moshi_urdu /training_data/processed/train-001-of-001.parquet"  # ~28 MB
ls -lh "/home/ubuntu/vs_code/moshi_urdu /urdu_tokenizer.model"                              # ~875 KB
ls "/home/ubuntu/vs_code/moshi_urdu /moshi-finetune/finetune.py"                           # exists
```

You also need:
- A HuggingFace account + access token (for downloading kyutai/moshiko-pytorch-bf16)
- A RunPod account with billing enabled
- (Optional) Wandb account for live training logs

---

## 1. Pick a GPU pod

| Tier | GPU | RunPod price | Suitable for |
|---|---|---|---|
| **Budget**       | 2× L40S 48GB     | ~$1.40/hr | Tight — may need batch=2 |
| **Recommended**  | **2× A100 80GB**  | **~$3.60/hr** | POC, 5 epochs in ~3 hrs |
| **Faster**       | 4× A100 80GB     | ~$7/hr  | POC, 5 epochs in ~1.5 hrs |
| **Premium**      | 4× H100 80GB     | ~$14/hr | Real run, 10+ epochs |

Pod settings:
- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.1.1` (or any pytorch+cuda12 image)
- Volume disk: **100 GB** (model checkpoints are big — float32 7.7B model is 30 GB, plus checkpoints)
- Container disk: 30 GB

Deploy and wait until "Running."

---

## 2. Bundle and upload the data

### 2.1 On LOCAL machine

```bash
cd "/home/ubuntu/vs_code/moshi_urdu "
tar -cf finetune_bundle.tar \
    moshi-finetune/ \
    runpod_finetune/ \
    urdu_tokenizer.model \
    training_data/processed/

# Verify size — should be ~30-35 MB
ls -lh finetune_bundle.tar

# Send via croc (peer-to-peer)
croc send finetune_bundle.tar
# Note the code printed (e.g. "abc-def-ghi")
```

### 2.2 On the POD

Install croc (one-time):
```bash
curl https://getcroc.schollz.com | bash
```

Receive the bundle:
```bash
cd /workspace
croc <CODE>
tar -xf finetune_bundle.tar

# Verify layout
ls /workspace/
#   moshi-finetune/  runpod_finetune/  urdu_tokenizer.model  training_data/

# Move parquet to a clean path
mkdir -p /workspace/processed
mv /workspace/training_data/processed/*.parquet /workspace/processed/
ls /workspace/processed/
#   train-001-of-001.parquet
```

---

## 3. Install dependencies

```bash
cd /workspace
bash runpod_finetune/setup.sh
```

This takes ~10-15 min (mostly downloading torch + deepspeed). If anything errors, you'll see it; just paste me the error.

When done, log into HuggingFace:
```bash
huggingface-cli login
# Paste your HF token (read access is enough)
```

---

## 4. Init the Moshi model for Urdu

```bash
bash runpod_finetune/run_init.sh
```

This will:
1. Download `kyutai/moshiko-pytorch-bf16` from HF (~15 GB, ~2-3 min on pod's fast network)
2. Re-init text embeddings (Gaussian noise except for IDs 0, 3, 32000)
3. Add Speaker B output heads (`--extend_modules_for_user_stream`)
4. Convert to MoshiForFinetuning class (DeepSpeed-compatible)
5. Save as float32 to `/workspace/init_models/moshiko-urdu-float32/` (~30 GB)

**Total time: ~5-10 min.**

Verify:
```bash
ls -lh /workspace/init_models/moshiko-urdu-float32/
# model.safetensors  (~30 GB)
# moshi_lm_kwargs.json
```

---

## 5. (Optional) Set up Weights & Biases for live training logs

```bash
pip install wandb
wandb login  # paste WANDB token
export USE_WANDB=1
```

---

## 6. Launch fine-tuning

### Default config (good for 2× A100 80GB, 5 epochs):
```bash
cd /workspace
bash runpod_finetune/run_finetune.sh
```

### Override config via env vars:
```bash
NUM_GPUS=4 NUM_EPOCHS=10 PER_DEVICE_BATCH=4 GRAD_ACCUM=4 \
    bash runpod_finetune/run_finetune.sh
```

### What you'll see during training:
```
=== Fine-tune config ===
  GPUs: 2
  Effective batch: 64
  Epochs: 5
=========================

[loading parquet, preprocessing...]
[init deepspeed zero 3...]
[step 1/105]  loss=8.234  text_loss=6.12  audio_loss=8.43  lr=1e-6  elapsed=12s
[step 2/105]  loss=7.891  text_loss=5.98  audio_loss=8.12  lr=2e-6  elapsed=23s
...
[saving checkpoint at step 100 → output/moshi-urdu-finetuned/step_100/]
...
```

**Loss should decrease steadily.** If it spikes or NaNs, check:
- VRAM not saturated (try lower batch)
- LR too high (rare with default warmup)
- Data malformed (we already verified)

### Estimated training time
- **2× A100 80GB, 5 epochs:** ~2-3 hrs
- **4× A100 80GB, 5 epochs:** ~1-1.5 hrs
- **4× H100 80GB, 10 epochs:** ~2-3 hrs

---

## 7. Convert checkpoint for inference

DeepSpeed saves a "state" format that needs conversion:

```bash
cd /workspace/moshi-finetune

# Find the latest checkpoint
ls /workspace/output/moshi-urdu-finetuned/
# step_100  step_200  step_300  ...

LATEST=$(ls /workspace/output/moshi-urdu-finetuned/ | grep "^step_" | sort -t_ -k2 -n | tail -1)
echo "Converting $LATEST..."

python -m tools.zero_to_fp32 \
    /workspace/output/moshi-urdu-finetuned/$LATEST \
    /workspace/output/moshi-urdu-finetuned/${LATEST}_fp32 \
    --moshi_lm_kwargs_path /workspace/init_models/moshiko-urdu-float32/moshi_lm_kwargs.json
```

---

## 8. Test the model — generate Urdu speech

```bash
cd /workspace
MODEL_DIR=/workspace/output/moshi-urdu-finetuned/${LATEST}_fp32 \
    bash runpod_finetune/run_generate.sh
```

This feeds the first 10 seconds of held-out audio to the model and asks it to continue for 20 seconds. Outputs:
- WAV files of generated speech
- JSON of generated text tokens (decode with urdu_tokenizer.model)

Listen to a sample:
```bash
ls /workspace/output/moshi-urdu-finetuned/${LATEST}_fp32/continuations/
# Download the .wav files via croc back to local
```

---

## 9. Download trained model + samples back to local

The fine-tuned model is BIG (~30 GB). For now just download:
- The fp32 safetensors (the trained model, ~30 GB) — only if you want it locally
- A few generated .wav samples (small) — to listen and validate

```bash
# On POD
cd /workspace/output/moshi-urdu-finetuned
tar -cf samples.tar ${LATEST}_fp32/continuations/
croc send samples.tar

# Optionally also send the model (large):
tar -cf urdu_model.tar ${LATEST}_fp32/
croc send urdu_model.tar
```

---

## 10. Terminate the pod

**Critical step** — don't leave the pod running, you keep paying.

RunPod console → your pod → Stop → Terminate.

---

## Cost summary (estimated)

| Stage | Time on 2× A100 | Time on 4× A100 | Cost @ A100 |
|---|---|---|---|
| Setup (install + HF login) | 15 min | 15 min | $1-2 |
| Init Moshi for Urdu | 10 min | 10 min | $0.60-1 |
| Fine-tune (5 epochs) | 2-3 hr | 1-1.5 hr | $7-15 |
| Generate samples | 5-10 min | 5-10 min | $0.30-1 |
| **Total POC** | **~3 hr** | **~2 hr** | **~$10-20** |

---

## Troubleshooting

**OOM (out of memory):**
- Lower `PER_DEVICE_BATCH` to 2 or 1
- Increase `GRAD_ACCUM` to keep effective batch the same
- Verify `--activation_checkpointing` is on (it is by default)

**NaN loss:**
- Lower learning rate (`--tempformer_learning_rate 5e-7 --depformer_learning_rate 5e-7`)
- Restart from a saved checkpoint instead of init

**HF download fails / 401:**
- Re-run `huggingface-cli login` with a valid read token

**deepspeed C extension build fails:**
- Pod must have CUDA dev toolkit. Use template `runpod/pytorch:*-cuda12.1*` not the slim variant.

**Slow training (way over estimate):**
- Check `nvidia-smi` — are all GPUs at 90%+ utilization? If not, increase batch or lower grad_accum.
