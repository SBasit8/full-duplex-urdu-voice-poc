  # WhisperX on RunPod — Step by Step

## 1. Rent a RunPod GPU Pod
- Go to https://www.runpod.io/console/pods
- Click "Deploy" → pick a template (e.g. `runpod/pytorch:2.1.0-py3.10-cuda11.8.0`)
- GPU: **A10G 24GB** ($0.30/hr spot) or **RTX 4090** ($0.44/hr) — either works
- Storage: set **Volume** to at least 20 GB (for data + model weights)
- Deploy and wait for it to start
- Click "Connect" → use SSH or the web terminal

## 2. Upload data to the pod
From YOUR LOCAL machine (this one), run:

```bash
# Replace <POD_IP> and <SSH_PORT> with your RunPod pod's SSH details
scp -P <SSH_PORT> -r "/home/ubuntu/vs_code/moshi_urdu /training_data/stereo_dialogues_cv" root@<POD_IP>:/workspace/stereo_dialogues_cv

# Upload the scripts
scp -P <SSH_PORT> -r "/home/ubuntu/vs_code/moshi_urdu /runpod_whisperx/" root@<POD_IP>:/workspace/runpod_whisperx
```

## 3. SSH into the pod and install
```bash
ssh -p <SSH_PORT> root@<POD_IP>
cd /workspace/runpod_whisperx
bash setup.sh
```

## 4. Run the bulk processing
```bash
cd /workspace
python runpod_whisperx/process_all.py \
    --input_dir ./stereo_dialogues_cv \
    --output_dir ./word_timestamps \
    --model large-v3 \
    --language ur \
    --batch_size 16 \
    --skip_existing
```

- **--skip_existing** lets you resume if it crashes — already-done files are skipped
- **--batch_size 16** is good for 24GB VRAM; lower to 8 if you see OOM
- Expected time: ~2-3 hours for 1,367 dialogues (38 hr audio)
- Progress is printed per file

## 5. Download results back to this machine
From YOUR LOCAL machine:
```bash
scp -P <SSH_PORT> -r root@<POD_IP>:/workspace/word_timestamps "/home/ubuntu/vs_code/moshi_urdu /training_data/word_timestamps"
```

## 6. Terminate the pod
Go to RunPod console → Stop/Terminate the pod to stop billing.

## Output format
Each dialogue gets a matching JSON:
```
word_timestamps/
├── <pair_folder>/
│   ├── dlg_0000.json
│   ├── dlg_0001.json
│   └── ...
```

Each JSON is an array of word-level entries sorted by time:
```json
[
  {"speaker": "A", "word": "سلام", "start": 0.46, "end": 0.82},
  {"speaker": "B", "word": "وعلیکم", "start": 1.20, "end": 1.65},
  ...
]
```

This matches the moshi-finetune `data/spokenwoz_sample/text/*.json` format exactly.
