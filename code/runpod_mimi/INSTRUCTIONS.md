# Mimi Audio Tokenization on RunPod — Step by Step

## 0. What this does
Runs the Mimi encoder on 1,367 stereo dialogues (38.33 hr of audio) →
produces `.npz` files with shape `[8, T]` per speaker (8-level RVQ @ 12.5Hz).
Expected time: ~30-60 min on an A10G 24GB, ~20 min on an RTX 4090.

## 1. Rent a RunPod GPU Pod
- https://www.runpod.io/console/pods → Deploy
- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.1.1` (or similar)
- GPU: **A10G 24GB** is enough. RTX 4090 or A40 also fine.
- Volume: **30 GB** (audio data + model weights)
- Deploy, wait for Ready.

## 2. Upload data + scripts from this LOCAL machine

**Important:** `stereo_dialogues_flat/` on local is just symlinks; you need to upload the REAL
audio which lives in `stereo_dialogues_cv/` (13 GB, nested as `<pair>/dlg_XXXX.wav`). Flatten
on the pod after upload.

Fastest path — use **croc** (same tool as WhisperX upload):

```bash
# === ON LOCAL MACHINE ===
cd "/home/ubuntu/vs_code/moshi_urdu /training_data"
tar -cf stereo_cv.tar stereo_dialogues_cv/
croc send stereo_cv.tar   # prints a code like "able-cupid-robot"

# Also send the moshi-finetune repo + setup.sh (small, <10 MB)
cd "/home/ubuntu/vs_code/moshi_urdu "
tar -cf mft.tar moshi-finetune/ runpod_mimi/
croc send mft.tar
```

```bash
# === ON POD ===
cd /workspace
croc <CODE_FOR_STEREO_CV>    # downloads stereo_cv.tar
tar -xf stereo_cv.tar
croc <CODE_FOR_MFT>          # downloads mft.tar
tar -xf mft.tar

# Flatten the nested dir into flat <pair>__<dlg>.wav layout
python3 -c "
from pathlib import Path
src = Path('/workspace/stereo_dialogues_cv')
dst = Path('/workspace/stereo_dialogues_flat'); dst.mkdir(exist_ok=True)
n = 0
for p in src.rglob('dlg_*.wav'):
    (dst / f'{p.parent.name}__{p.stem}.wav').symlink_to(p.resolve())
    n += 1
print(f'flattened {n}')
"
```

## 3. Install deps on the pod
```bash
cd /workspace
bash runpod_mimi/setup.sh
```
First run downloads Mimi weights from HF (~600 MB) — allow a couple minutes.

## 4. Run the audio tokenization
```bash
cd /workspace/moshi-finetune
python -m tools.tokenize_audio \
    --audio_dir /workspace/stereo_dialogues_flat \
    --output_dir /workspace/tokenized_audio \
    --num_workers 1 \
    --resume
```

- Single worker = 1 GPU. If pod has multiple GPUs, raise `--num_workers`.
- `--resume` skips already-tokenized files (safe to restart if it crashes).
- Expected: ~30-60 min for 1,367 files on A10G.
- Each output `.npz` is ~50-200 KB. Total output ~200 MB.

## 5. Download results back to local machine
From local terminal:
```bash
# Option A: scp (if direct SSH works)
scp -P <SSH_PORT> -r root@<POD_IP>:/workspace/tokenized_audio \
    "/home/ubuntu/vs_code/moshi_urdu /training_data/tokenized_audio"

# Option B: croc
# On pod: cd /workspace && tar -cf tok_audio.tar tokenized_audio/ && croc send tok_audio.tar
# On local: cd training_data/ && croc <CODE> && tar -xf tok_audio.tar
```

## 6. Terminate the pod
RunPod console → Stop / Terminate. Otherwise you'll keep paying.

## Output format (sanity check)
```python
>>> import numpy as np
>>> npz = np.load("tokenized_audio/<pair>__dlg_0000.npz")
>>> npz["A"].shape   # (8, T)  — 8 RVQ levels, T frames at 12.5Hz
>>> npz["B"].shape   # same
```
T should match audio duration × 12.5. For a 60s dialogue → T ≈ 750.
