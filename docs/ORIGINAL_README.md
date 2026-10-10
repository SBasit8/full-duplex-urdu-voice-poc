# Urdu Moshi — POC

End-to-end proof-of-concept for fine-tuning [Kyutai's Moshi](https://kyutai.org/Moshi.pdf) full-duplex spoken dialogue model on Urdu, using the [moshi-finetune](https://github.com/nu-dialogue/moshi-finetune) framework as the training scaffold.

**Status:** POC complete. Architecture validated. Training data scale insufficient for production-quality output (expected — see [Findings](#findings)).

**Trained model:** [`S-basit/moshi-urdu-poc`](https://huggingface.co/S-basit/moshi-urdu-poc) (private HuggingFace repo, 16 GB)

---

## What this project is

We took Kyutai's English/French Moshi base (`kyutai/moshiko-pytorch-bf16`) and adapted it to Urdu by:

1. **Building an Urdu SentencePiece tokenizer** (32K vocab — must match Moshi's vocab size exactly)
2. **Synthesizing dialogue data** from Common Voice 25.0 Urdu (1,367 stereo dialogues, 38.3 hours)
3. **Re-initializing the model's text embeddings** so it learns Urdu vocabulary (audio embeddings retained — Mimi is language-agnostic)
4. **Fine-tuning** with DeepSpeed Zero 3 on 2× H100 80GB
5. **Inference** to produce sample continuations + decoded WAV/text outputs

The result is a model that produces Urdu-phoneme-shaped audio and emits real Urdu words in its text stream — but without coherent sentence structure (38 hrs is ~0.05% of what J-Moshi used).

## Architecture: original Moshi vs. our Urdu Moshi

| Component | Original Moshi | Our Urdu Moshi |
|---|---|---|
| Audio encoder/decoder (Mimi) | Frozen, language-agnostic | Same (universal) |
| Temporal Transformer (~7B params) | Trained on ~7M hrs English/French | Fine-tuned from Kyutai's checkpoint |
| Depth Transformer (~700M) | Trained on ~7M hrs | Fine-tuned from Kyutai's |
| Text tokenizer | Kyutai's English SPM (32K vocab) | **Custom Urdu SPM (32K vocab)** |
| Text embedding table | Kyutai's | **Re-initialized**, learned from scratch on Urdu |
| Audio output heads | 8 (Moshi-only) | **16** (Moshi + user) — dual-stream dialogue |
| Training data | 7M hrs general + 100s hrs dialogue | **38 hrs synthetic Urdu dialogue** |
| Training compute | Thousands of GPU-hours | **~6 GPU-hours** (2× H100 × 3 hrs) |

## Pipeline overview

```
   Common Voice 25.0 Urdu (raw MP3)
            │
            │  build_training_data.py
            │  (filter, exclude dev/test speakers)
            ▼
   53.3 hrs of clean clips × 80 speakers
            │
            │  stitch_cv.py
            │  (group by client_id, pack to ~120 sec)
            ▼
   2,264 mono segments (24 kHz, 16-bit WAV)
            │
            │  make_stereo_dialogues.py
            │  (gender-aware speaker pairing,
            │   alternating turns, L=A R=B stereo)
            ▼
   1,367 stereo dialogues = 38.3 hr training audio
            │
            │  runpod_whisperx/  (GPU pod, ~$0.40)
            │  (large-v3 + wav2vec2 alignment)
            ▼
   Word-level timestamp JSONs (1,367 files)
            │
            │  moshi-finetune/tools/tokenize_text.py
            │  (with our Urdu SPM)
            ▼  ┌───────────────────────────────┐
              │  moshi-finetune/tools/        │
              │  tokenize_audio.py (Mimi)    │ runpod_mimi/  (~$0.05)
              ▼  └───────────────────────────────┘
   1,362 (text + audio) token .npz pairs
            │
            │  moshi-finetune/tools/prepare_dataset.py
            ▼
   train-001-of-001.parquet (28 MB)
            │
            │  moshi-finetune/tools/init_moshi_for_ft.py
            │  --init_text_embeddings
            │  --extend_modules_for_user_stream
            ▼
   moshiko-urdu-float32 init checkpoint (32 GB)
            │
            │  moshi-finetune/finetune.py + DeepSpeed Zero 3
            │  (5 epochs, 2× H100 80GB, ~30 min training, ~$5)
            ▼
   model.safetensors (16 GB) → uploaded to HF
            │
            │  moshi-finetune/generate.py + decode_tokens.py
            ▼
   Sample WAVs + Urdu transcripts (in samples/)
```

To hear how the audio transforms across the pipeline, open the clips in [`dataset_samples/`](dataset_samples/): one raw Common Voice MP3 → one merged single-speaker segment → two stitched stereo dialogues (`MF` = male/female pair, `XX` = leftover/unknown-gender pair).

## Repository structure

```
moshi-urdu/
├── README.md                              # this file
├── docs/
│   ├── architecture.md                    # detailed architecture deep-dive
│   ├── pipeline.md                        # step-by-step build + training guide
│   └── findings.md                        # results + decoupling analysis
│
├── urdu_tokenizer.model                   # trained Urdu SPM (32K vocab)
├── urdu_tokenizer.vocab                   # human-readable vocab
├── train_tokenizer.py                     # script that built it
├── verify_tokenizer.py                    # sanity-check tokenizer
│
├── download_urdu_nashtiliq.py             # Urdu Dolly text downloader
├── build_training_data.py                 # CV filter + manifest builder
├── stitch_cv.py                           # group same-speaker clips → segments
├── make_stereo_dialogues.py               # synthetic L/R stereo dialogues
│
├── runpod_whisperx/                       # GPU pod scripts: word-level timestamping
│   ├── INSTRUCTIONS.md
│   ├── process_all.py
│   └── setup.sh
│
├── runpod_mimi/                           # GPU pod scripts: Mimi audio tokenization
│   ├── INSTRUCTIONS.md
│   └── setup.sh
│
├── runpod_finetune/                       # GPU pod scripts: training + inference
│   ├── INSTRUCTIONS.md
│   ├── setup.sh                           # install moshi + deepspeed
│   ├── run_init.sh                        # init Moshi for Urdu
│   ├── run_finetune.sh                    # main training launcher
│   ├── run_generate.sh                    # full eval generation
│   ├── run_generate_few.sh                # quick 10-sample generation
│   ├── merge_shards.py                    # consolidate sharded safetensors
│   └── test_my_voice.py                   # custom-prompt inference helper
│
├── moshi-finetune/                        # patched fork (4 upstream bug fixes)
│   ├── finetune.py                        # main training loop
│   ├── generate.py                        # inference / continuation
│   ├── tools/                             # tokenize_audio/text, prepare_dataset, decode_tokens, etc.
│   ├── models/, utils/, ds_configs/
│   └── ...
│
├── dataset_samples/                       # one audio per pipeline phase (raw → merged → stitched)
│   ├── 01_raw_cv/                         # original Common Voice mp3
│   ├── 02_segments_merged/                # same-speaker clips packed to ~120 s
│   └── 03_stitched_dialogues/             # stereo MF + XX pair examples
│
├── training_data/
│   └── processed/
│       └── train-001-of-001.parquet       # 1,362 tokenized dialogues, ready for training
│
└── trained_models/
    ├── samples/                           # 10 generated samples (WAV + Urdu text)
    │   ├── 0.wav  0.txt
    │   ├── 1.wav  1.txt
    │   └── ...
    ├── training2.log                      # full training output (loss curves)
    └── generate.log                       # inference output
```

## Quick start: reproduce or iterate

### Prerequisites

- Linux machine for data prep (CPU OK)
- HuggingFace account with read+write tokens
- RunPod account (or any GPU host with H100/A100/L40S)
- Python 3.11+, ffmpeg installed locally

### 1. Clone

```bash
git clone https://github.com/siglance/moshi-urdu.git
cd moshi-urdu
```

### 2. Install local dependencies

```bash
pip install sentencepiece numpy pandas datasets huggingface_hub safetensors soundfile
```

### 3. Generate or get the data

You have two paths:

**Path A — Use the included parquet (skip data prep):**

`training_data/processed/train-001-of-001.parquet` is already in the repo. Skip to step 4.

**Path B — Re-build from raw Common Voice:**

```bash
# Download Common Voice 25.0 Urdu manually from https://commonvoice.mozilla.org
# Place in 1774212483838-cv-corpus-25.0-2026-03-09-ur/

# Build train manifest
python build_training_data.py

# Stitch same-speaker clips into segments
python stitch_cv.py

# Make stereo dialogues
python make_stereo_dialogues.py

# Word-level timestamps (needs GPU — see runpod_whisperx/INSTRUCTIONS.md)
# Audio tokenization (needs GPU — see runpod_mimi/INSTRUCTIONS.md)
# Text tokenization + parquet build (CPU)
cd moshi-finetune
python -m tools.tokenize_text --word_transcript_dir ../training_data/word_timestamps_flat \
    --output_dir ../training_data/tokenized_text \
    --text_tokenizer_path ../urdu_tokenizer.model \
    --num_workers 8
python -m tools.prepare_dataset \
    --tokenized_text_dir ../training_data/tokenized_text \
    --tokenized_audio_dir ../training_data/tokenized_audio \
    --output_prefix ../training_data/processed/train
```

### 4. Train on a GPU pod

See [`runpod_finetune/INSTRUCTIONS.md`](runpod_finetune/INSTRUCTIONS.md) for the full step-by-step on RunPod, including all the env vars, patches, and config tweaks we discovered during the POC.

Short version on a 2× H100 80GB pod:

```bash
# Upload code + parquet + tokenizer (use rsync or croc)
# Then on the pod:
bash runpod_finetune/setup.sh
huggingface-cli login
bash runpod_finetune/run_init.sh

export PER_DEVICE_BATCH=2 GRAD_ACCUM=16 SAVE_STEPS=200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash runpod_finetune/run_finetune.sh
```

Total training time: ~30 min for 5 epochs on 38 hrs of data.

### 5. Inference

Pick a few prompts from your parquet and generate continuations:

```bash
NUM_EXAMPLES=10 MODEL_DIR=/workspace/output/moshi-urdu-finetuned/step_110 \
    bash runpod_finetune/run_generate_few.sh

cd moshi-finetune
python -m tools.decode_tokens \
    --tokens_dir /workspace/output/moshi-urdu-finetuned/step_110/continuations/generated_tokens \
    --output_dir /workspace/decode_output \
    --text_tokenizer_path /workspace/urdu_tokenizer.model \
    --num_workers 1
```

## Findings

See [`docs/findings.md`](docs/findings.md) for the detailed POC writeup. Headlines:

### What worked

- **Pipeline integrity** — end-to-end 38 hr → trained model → samples without architectural changes
- **Tokenizer compatibility** — 32K Urdu SPM successfully replaced 32K English SPM
- **Text embedding learning** — real Urdu words emerged in text stream (e.g., پنجاب, خوشی, زندگی, گوادر, موٹرسائیکل)
- **Audio output** — decoded to speech-like signal (not noise)
- **Dual stream** — L/R channels preserved; both A and B generated independently

### What was rough

- **Audio coherence** — speech-like but not recognizable as fluent Urdu
- **Text coherence** — real words but no sentence structure
- **Text-audio alignment** — the two streams don't say the same thing yet (the **decoupling phenomenon**)
- **Voice consistency** — voice shifts within a single 20-sec generation

### Why this is consistent with expectations

Comparison to [J-Moshi](https://nu-dialogue.github.io/j-moshi):
- J-Moshi: 69,000 hrs of data, 4,608 GPU-hours of training
- This POC: 38 hrs of data, **6 GPU-hours**

That's ~0.02% of J-Moshi's compute. The fact that Urdu words emerge at all is itself a strong signal that the architecture transfers. With 200+ hrs of real conversational data and 30+ epochs, the same architecture should produce demoable output.

### The decoupling issue (architectural, recoverable)

In Moshi's design, the Depth Transformer predicts text first, then conditions audio on text. With low-quality text predictions early in training, the audio side learns to ignore the text — they decouple. After enough training, when text is informative, the streams re-couple.

We're stuck in the "decoupled" phase due to compute budget. More training would resolve this without architectural changes. See [`docs/findings.md`](docs/findings.md) for the full analysis.

## Costs

| Phase | Cost |
|---|---|
| Tokenizer training | $0 (CPU) |
| Data preprocessing | $0 (CPU) |
| WhisperX timestamping | ~$0.40 (1× A10G × 1.2 hr) |
| Mimi audio tokenization | ~$0.05 (1× A10G × 5 min) |
| Moshi init + fine-tune (with all retries) | ~$15–20 (2× H100 × 3 hr + setup overhead) |
| Sample generation + HF upload | ~$1.50 (residual pod time) |
| **Total POC** | **~$17–22** |

A clean rerun with all bugs known would be ~$5–7 for one full training cycle.

## Next steps

| Improvement | Cost | Expected gain |
|---|---|---|
| Train 15-30 epochs on same data | $30-60 | Resolves decoupling; better text quality |
| Higher text loss weight | $30 | Faster vocabulary convergence |
| Pre-trained Urdu embeddings init | $15 + eng | Skip ~10 epochs of vocab learning |
| LoRA on text path only | $10/iter | Cheap iteration on tokenizer experiments |
| **Acquire 200+ hrs real Urdu conversation** | major data work | Fundamentally better quality |
| **Full retrain on real data, 30+ epochs** | $500-2000 | Demoable Urdu Moshi |

See [`docs/findings.md`](docs/findings.md) for detailed reasoning behind each.

## Patches we made to upstream moshi-finetune

The `moshi-finetune/` directory in this repo is a patched fork. Diffs vs. [the upstream commit we forked from](https://github.com/nu-dialogue/moshi-finetune):

1. **`finetune.py`** — fixed `args.with_tracking` AttributeError (line ~298)
2. **`finetune.py`** — replaced `accelerator.save_state(output_dir)` with `accelerator.save_model(moshi_lm, output_dir, safe_serialization=True)` (lines ~954, ~961). Avoids saving 70 GB optimizer states; saves only ~16 GB model weights, fits in disk budget.
3. **`utils/data.py`** — replaced `np.concat` with `np.concatenate` (older numpy compatibility)
4. **`tools/prepare_dataset.py`** — same `np.concat` fix
5. **`tools/tokenize_text.py`** — added `--text_tokenizer_path` arg for local SPM file
6. **`tools/tokenize_text.py`** — wrapped tokenize_and_pad_text in try/except so a single failing dialogue doesn't kill the multiprocessing worker
7. **`tools/decode_tokens.py`** — added `--text_tokenizer_path` arg, enabled saving text transcripts (`text` was previously commented out)
8. **`ds_configs/zero3-fp16-warmlr-act_ckpt.json`** — set `stage3_gather_16bit_weights_on_model_save: true` for cleaner save format

If anyone upstreams this, file PRs to nu-dialogue/moshi-finetune.

## License

Code: MIT (see [LICENSE](LICENSE))
moshi-finetune fork: same license as upstream (Apache-2.0, see `moshi-finetune/LICENSE`)
Common Voice data: not included in repo (download per Mozilla terms)
Trained model on HF: private siglance org access only

## Credits

- [Kyutai](https://kyutai.org) — Moshi model + Mimi codec + base weights
- [nu-dialogue/moshi-finetune](https://github.com/nu-dialogue/moshi-finetune) — fine-tuning framework
- [Mozilla Common Voice](https://commonvoice.mozilla.org) — Urdu speech corpus
- [WhisperX](https://github.com/m-bain/whisperX) — word-level alignment
