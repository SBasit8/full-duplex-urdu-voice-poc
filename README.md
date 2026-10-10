# Urdu Moshi — Full-Duplex Voice Model Proof-of-Concept

Fine-tuning Kyutai's [Moshi](https://arxiv.org/abs/2410.00037) (+ Mimi codec) into a full-duplex
**Urdu** conversational voice model. Status: **proof-of-concept, paused pending data** — the
pipeline runs end-to-end and the model emits real Urdu words, but at 38.33 hours of training
audio it has nowhere near the data needed for coherent speech. The full writeup, with every
number traced to its source file, is in [`docs/research_report.html`](docs/research_report.html).

## What's in this repo

This is the **code, small samples, logs, and research** — not the full dataset or model weights,
which are too large for plain Git (the checkpoint alone is ~16 GB; the training audio is ~38
hours). Where to get those is in [Large artifacts](#large-artifacts-not-in-this-repo) below.

| Folder | Contents |
|---|---|
| `docs/` | Project handoff docs (architecture, pipeline, datasets, training, results, status) + the full research report (`research_report.html`) |
| `code/` | All pipeline scripts (tokenizer, data prep, stereo-dialogue builder), the patched `moshi-finetune` fork, and the RunPod job scripts (WhisperX, Mimi tokenization, fine-tune) |
| `samples/generated/` | The 10 generated continuations (WAV + decoded Urdu text) from checkpoint `step_110` |
| `samples/dataset_samples/` | One audio example per pipeline phase: raw Common Voice clip → merged segment → stitched stereo dialogue |
| `tokenizer/` | The trained Urdu SentencePiece tokenizer (32k vocab) |
| `logs/` | Raw build and training logs (`training2.log` has all 110 logged optimizer steps) |
| `research/` | Research notes, the four archived reference papers, and the full bibliography |

## Key numbers

- **Data:** Common Voice v25 Urdu (294.65 h raw) → filtered 53.33 h → **1,367 stereo dialogues /
  38.33 h** → 1,362 rows in the final training parquet.
- **Training:** 5 epochs / 110 steps, 2 GPUs, DeepSpeed Zero-3. Final loss **5.39** (text 3.72,
  audio 1.67). Total cost: **$17–22**.
- **Result:** The model produces real, correctly-inflected Urdu words — but no coherent sentence
  structure. Consistent with using ~0.02–0.13% of the data/compute of the nearest published
  precedent, [J-Moshi](https://arxiv.org/abs/2506.02979) (69,000 h / ~4,600 GPU-hours).

Full detail, including a sourced background section on where Moshi itself (independent of this
Urdu work) falls short of natural conversational voice, is in the research report.

## Large artifacts not in this repo

| Artifact | Size | Where |
|---|---|---|
| Trained checkpoint (`moshi-urdu-step110`) | ~16 GB | Private HuggingFace repo `S-basit/moshi-urdu-poc` |
| Full stereo dialogue dataset | ~13 GB (38.33 h) | Not yet published — regenerable from Common Voice via `code/scripts/make_stereo_dialogues.py` |
| Common Voice 25.0 Urdu (raw source) | 6.6 GB | [commonvoice.mozilla.org](https://commonvoice.mozilla.org) |
| Base model `kyutai/moshiko-pytorch-bf16` | — | Public on HuggingFace |

## Reproduce

```bash
# 0–4, local (CPU): tokenizer + corpus + stereo dialogues
python code/scripts/train_tokenizer.py
python code/scripts/build_training_data.py
python code/scripts/make_stereo_dialogues.py

# 5–6, GPU pod: word timestamps (WhisperX) + Mimi audio tokenization
# see code/runpod_whisperx/INSTRUCTIONS.md and code/runpod_mimi/INSTRUCTIONS.md

# 7–8, local (CPU): text tokenization + parquet assembly
# see code/moshi-finetune/tools/tokenize_text.py and prepare_dataset.py

# 9–11, GPU pod (2x A100/H100): init, fine-tune, generate
# see code/runpod_finetune/INSTRUCTIONS.md
```

## Status

Paused by design after the proof-of-concept. The next step is **not** more training on the same
data — it's acquiring 200+ hours of real, channel-separated conversational Urdu. See
`docs/09_STATUS_AND_MISSING.md` and `docs/12_WORK_SUMMARY_worked-didnt-pivots.md` for the full
account of what worked, what didn't, and why.
