# 01 — Project Overview

| | |
|---|---|
| **What** | Fine-tune Kyutai **Moshi** (full-duplex speech LM) + **Mimi** codec into an **Urdu full-duplex conversational voice model** |
| **Status** | **Proof-of-concept complete** — pipeline runs end-to-end; model under-trained (38 h data) |
| **Org** | Siglance (GitHub repo planned, not yet created) |
| **Source folder** | `/home/ubuntu/vs_code/moshi_urdu ` (⚠️ trailing space in the name) |
| **Base model** | `kyutai/moshiko-pytorch-bf16` → `moshiko-urdu-float32` → `moshi-urdu-step110` |
| **Framework** | `nu-dialogue/moshi-finetune` (patched fork) |
| **Trained checkpoint** | HF **https://huggingface.co/S-basit/moshi-urdu-poc** (private, ~16 GB, ✅ confirmed present); local copy `weights/trained_models/` |
| **Compute** | RunPod cloud GPUs (2× A100/H100 for finetune) |
| **Snapshot date** | 2026-08-27 |

## The idea in three sentences
No Urdu full-duplex conversational voice model exists. This project adapts Moshi — which listens and
speaks simultaneously via **separated stereo streams** (L=agent, R=user) — to Urdu by training a new
Urdu tokenizer, synthesizing stereo two-speaker dialogues from Common Voice, tokenizing them with Mimi,
and fine-tuning Moshi's language model (Mimi stays frozen). The POC proved the whole pipeline works and
that Moshi/Mimi accept Urdu; the model emits real Urdu words but not coherent sentences because it saw
only **38 hours** vs the thousands needed.

## What's in this handoff
Real, independent file copies of everything — code, weights, **datasets**, research + papers, demo
media; envs as package lists. No hardlinks/symlinks; drive-ready. (Redundant `.tar` duplicates + absolute
symlink indexes were stripped — see `datasets/MANIFEST_EXCLUDED.md`.)

| Section | Doc | Folder |
|---|---|---|
| Research & approach (full vs half duplex, Moshi/Mimi, roadmap) | `02` | `research/` |
| Architecture & 11-stage pipeline | `03` | — |
| **Full-duplex** (stereo dialogues + dual-stream init) | `04` | — |
| Datasets (inventory, splits, how used) | `05` | `datasets/` (hardlinked) |
| Data-pipeline scripts | `06` | `code/scripts/`, `code/runpod_*` |
| Training & fine-tune | `07` | `code/moshi-finetune/`, `code/runpod_finetune/` |
| Model weights & results | `08` | `weights/` |
| Status / planned / missing | `09` | — |
| Environment & setup | `10` | `env/` |

## Key numbers
- **Data:** Common Voice v25 Urdu (294 h raw) → filtered 53 h → **1,367 stereo dialogues / 38.33 h** → 1,362 in the training parquet. Tokenizer trained on `ur.txt` (5.8 G), vocab **32,000**.
- **Training:** 5 epochs / **110 steps**, 2 GPUs, DeepSpeed Zero-3, final loss ≈5.4 (text 3.7).
- **Result:** Urdu words, no coherent syntax (expected at this scale).

## 🚨 First things to check
1. **Reconcile the checkpoint** — local `step110` is 3.97 GB fp16 (looks partial) vs 16 GB on HF. Pull the HF copy + keep the init `moshi_lm_kwargs.json` (needed to load it). (`08`)
2. **Rotate the 2 exposed HuggingFace tokens** found in `summary.txt` (redacted in the copy). (`SECRETS/`)
3. **Recreate the missing `docs/`** the README links to (architecture/pipeline/findings). (`09`)
4. Datasets are **hardlinks** — if you move this folder to another disk, use `cp -rL`/zip to materialize the real bytes.
