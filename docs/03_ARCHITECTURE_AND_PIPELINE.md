# 03 — Architecture & End-to-End Pipeline

The POC fine-tunes Kyutai **Moshi** (`kyutai/moshiko-pytorch-bf16`) into an Urdu **full-duplex**
model using the **`nu-dialogue/moshi-finetune`** fork (patched, vendored in `code/moshi-finetune/`).
Because no real Urdu conversation data exists, dialogues are **synthesized** by pairing Common Voice
speakers into stereo (L = speaker A, R = speaker B).

> Git: the whole POC is one snapshot — 2 commits (`5aaf78c Initial commit: Urdu Moshi POC`,
> `8bff55b Add dataset_samples/`). Not incrementally developed.

---

## The pipeline (11 stages, in order)

Every stage names the exact file (all under `code/`), what it does, inputs→outputs, and the command.

| # | Stage | Script | In → Out | Run |
|---|---|---|---|---|
| 0 | Text corpus / aux text | `scripts/download_urdu_nashtiliq.py` | HF `databricks-dolly-15k-ur` → `urdu_alpaca.jsonl`; bulk text = pre-existing `ur.txt` (6.1 GB) | `python download_urdu_nashtiliq.py` |
| 1 | **Tokenizer** | `scripts/train_tokenizer.py` (+ `verify_tokenizer.py`) | `ur.txt` → `urdu_tokenizer.model/.vocab` (SPM BPE, vocab 32000) | `python train_tokenizer.py` |
| 2 | Corpus assembly | `scripts/build_training_data.py` | 10hr TTS zips + Common Voice v25 → `training_data/common_voice/train_manifest.tsv` (+ `tts/manifest.tsv`, **unused**) | `python build_training_data.py` |
| 3 | Same-speaker stitch (mono) | `scripts/stitch_cv.py` | manifest → `stitched_cv/` (80 spk → 2,264 segs, 57.14 h) — **orphaned/demo, not in training path** | `python stitch_cv.py` |
| 4 | **Stereo dialogues (full-duplex)** | `scripts/make_stereo_dialogues.py` | manifest → `stereo_dialogues_cv/` (L=A,R=B; 40 pairs → **1,367 dialogues, 38.33 h**) | `python make_stereo_dialogues.py` |
| 5 | Word timestamps | `runpod_whisperx/process_all.py` | stereo WAVs → `word_timestamps/*.json` `{speaker,word,start,end}` | on pod, WhisperX `large-v3` `--language ur` |
| 6 | Mimi audio tokens | `moshi-finetune/tools/tokenize_audio.py` (via `runpod_mimi/`) | flattened stereo WAVs → `tokenized_audio/*.npz` `A,B` `[8,T]` @12.5 Hz | `python -m tools.tokenize_audio …` |
| 7 | Text tokens | `moshi-finetune/tools/tokenize_text.py` | word JSONs + Urdu SPM → `tokenized_text/*.npz` `[T]` (pad id 3, eot id 0) | `python -m tools.tokenize_text …` |
| 8 | Merge → parquet | `moshi-finetune/tools/prepare_dataset.py` | text+audio npz → `training_data/processed/train-001-of-001.parquet` (cols `dialogue_id`, `A[9,T]`, `B[9,T]`; **1,362** dialogues) | `python -m tools.prepare_dataset …` |
| 9 | Model init for Urdu | `moshi-finetune/tools/init_moshi_for_ft.py` (via `runpod_finetune/run_init.sh`) | `moshiko-pytorch-bf16` → `init_models/moshiko-urdu-float32/` (reinit text emb; `dep_q=16` user stream) | `bash run_init.sh` |
| 10 | **Fine-tune** | `moshi-finetune/finetune.py` (via `run_finetune.sh`, DeepSpeed Zero-3) | parquet + init → `output/…/step_110/` | `bash run_finetune.sh` |
| 11 | Convert → generate → decode | `zero_to_fp32` / `merge_shards.py` → `generate.py` → `decode_tokens.py` | checkpoint → `samples/{0..9}.wav`+`.txt` | `bash run_generate.sh` |

Full stage-by-stage detail (inputs, outputs, params, commands) is in `06_DATA_PIPELINE_SCRIPTS.md` and
`07_TRAINING_AND_FINETUNE.md`. The corresponding code is in `code/` and results in `weights/`.

---

## Moshi model (what's actually being trained)
- **Base:** `kyutai/moshiko-pytorch-bf16` — Temporal transformer (~7 B) + Depth transformer (~700 M) + **Mimi** codec (`tokenizer-e351c8d8-checkpoint125.safetensors`).
- Per **80 ms frame (12.5 Hz)** the model predicts **9 tokens**: 1 text + 8 Mimi RVQ audio codes.
- **Frozen:** Mimi encoder/decoder (always). **Trained:** Helium/Temporal + Depth transformers, the new Urdu text embeddings, and — via `--extend_modules_for_user_stream` (`dep_q=16`) — the **user-stream heads** that make it **dual-stream full-duplex** (predicts both agent and user audio). See `04_FULL_DUPLEX.md`.

## Dependency graph
```
ur.txt ─► train_tokenizer.py ─► urdu_tokenizer.model/.vocab
CV + TTS ─► build_training_data.py ─► common_voice/train_manifest.tsv
                                   ├─► stitch_cv.py ─► stitched_cv/ (mono, ORPHANED)
                                   └─► make_stereo_dialogues.py ─► stereo_dialogues_cv/ (L=A,R=B)
                                                                        ├─► whisperx/process_all.py ─► word_timestamps/
                                                                        └─► tools/tokenize_audio.py ─► tokenized_audio/ [8,T]
   word_timestamps/ + urdu_tokenizer.model ─► tools/tokenize_text.py ─► tokenized_text/ [T]
                          tokenized_text/ + tokenized_audio/ ─► tools/prepare_dataset.py ─► processed/*.parquet [9,T]
   moshiko-pytorch-bf16 ─► init_moshi_for_ft.py ─► init_models/moshiko-urdu-float32/
                          parquet + init ─► finetune.py (DeepSpeed Zero-3) ─► step_110/
                          ─► zero_to_fp32/merge_shards ─► step_110_fp32 ─► HF S-basit/moshi-urdu-poc
                          ─► generate.py ─► decode_tokens.py ─► samples/*.wav+*.txt
```

## Outcome
5 epochs / **110 steps** on 2 GPUs (~35 min per `training2.log`; README says H100/5.3 h — reconcile).
Final loss ≈ **5.4** (text ≈ 3.7, audio ≈ 1.65). The model emits **real Urdu words but not coherent
sentences** — the expected "decoupled" regime given 38 h of data vs J-Moshi's 69,000 h. See
`09_STATUS_AND_MISSING.md` and the alignment post-mortem in `02_RESEARCH_AND_APPROACH.md`.
