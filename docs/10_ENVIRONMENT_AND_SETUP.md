# 11 — Environment & Setup (recreate from scratch)

Two conda environments were used. They are **not copied** (13 GB total) — instead their exact package
lists are exported to `env/` so a new dev can rebuild them. Both are **Python 3.10** on **CUDA** GPUs.

| Env | Purpose | Python | Key packages | Export files |
|---|---|---|---|---|
| **`urdu-moshi`** | Moshi fine-tuning, Mimi tokenization, tokenizer training, generation | 3.10.12 | `torch==2.11.0`, `torchaudio==2.11.0`, `sentencepiece==0.2.1`, CUDA toolkit 13.0.2 / cuDNN 9.19 | `env/conda_list_urdu-moshi.txt` (225 pkgs), `env/pip_freeze_urdu-moshi.txt` (75) |
| **`whisperx-test`** | WhisperX forced-alignment → word timestamps for the audio | 3.10.20 | `whisperx==3.8.6`, `faster-whisper==1.2.1`, `ctranslate2==4.7.2`, `pyannote-audio==4.0.4`, `torch==2.8.0`, `transformers==4.57.6` | `env/conda_list_whisperx-test.txt` (161), `env/pip_freeze_whisperx-test.txt` (128) |

## Recreate the envs
```bash
# Moshi training env
conda create -n urdu-moshi python=3.10 -y
conda activate urdu-moshi
pip install -r env/pip_freeze_urdu-moshi.txt          # or install torch first, then moshi-finetune/requirements.txt
pip install -e code/moshi-finetune                    # the finetune package

# WhisperX env (word timestamps)
conda create -n whisperx-test python=3.10 -y
conda activate whisperx-test
pip install -r env/pip_freeze_whisperx-test.txt
```
> The `pip_freeze_*.txt` files pin exact versions (reproducible). The `conda_list_*.txt` files are the
> full conda inventory incl. CUDA libs, for reference. `code/moshi-finetune/requirements.txt` is the
> upstream dependency list for the finetune repo.

## GPU / compute
- Training and Mimi tokenization ran on **RunPod** cloud GPUs (see `code/runpod_*` job scripts and their `INSTRUCTIONS.md`). The trained checkpoint is a ~3 GPU-hour run on ~2 GPUs (per the research notes).
- The Mimi codec is always **frozen**; only the Moshi/Helium weights + text embeddings + tokenizer are trained.

## Where things are in this handoff
| Need | Path |
|---|---|
| Training/finetune + pipeline code | `code/` (see `07`, `06`) |
| Model weights + samples | `weights/trained_models/` (see `08`) |
| Tokenizer | `weights/tokenizer/` |
| Datasets (hardlinked) | `datasets/` (see `03`) |
| Demo audio/video | `demo_media/` |
| Research | `research/` (see `09`/`10`) |
| Env package lists | `env/` |
