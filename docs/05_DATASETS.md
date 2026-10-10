# 05 — Datasets (inventory, splits, how used)

All datasets are **real, independent file copies** in `datasets/` (no hardlinks/symlinks — drive-ready).
Redundant `.tar` duplicates and absolute-symlink index dirs were excluded; see
`datasets/MANIFEST_EXCLUDED.md` for exactly what was removed and the one-liners to regenerate it. Sizes
below are the real on-disk sizes.

> **The one irreplaceable item is the Common Voice corpus.** Everything under
> `stitched/stereo/tokenized/processed` is **regenerable** from it via the scripts in `code/`.

---

## 1. Inventory

| # | Dataset (`datasets/…`) | Source | Size | Count | Hours | Format | Role |
|---|---|---|---|---|---|---|---|
| 1 | `common_voice_urdu_v25/…/ur/` | Mozilla **Common Voice 25.0** Urdu | 6.6 G | 250,779 mp3 + 10 tsv | **294.65 h** (Σ `clip_durations.tsv`) | mp3 + tsv | ⭐ **SOURCE** raw corpus |
| 2 | `urdu_tts_10hr/` (zips) | purchased **10-hr Urdu TTS** corpus | 516 M | 37 zip | ~9.8 h | zipped webm+tsv | **SOURCE** (single-speaker; **unused** in final training) |
| 3 | `training_data/tts/` | unzip of #2 | 527 M | 5,311 webm | ~9.8 h | webm+tsv | staged, **unused downstream** |
| 4 | `training_data/common_voice/train_manifest.tsv` | filtered CV (via `build_training_data.py`) | 13 M | 51,279 rows | **53.33 h** (Σ `duration_ms`) | tsv → mp3 | DERIVED — filtered manifest |
| 5 | `training_data/stitched_cv/` | same-speaker packing | 8.7 G | 2,264 seg + json | ~57 h | **mono** 24 k WAV | DERIVED — **orphaned** (not in training path) |
| 6 | `training_data/stereo_dialogues_cv/` | A/B paired dialogues | 13 G | 1,367 wav + json | **38.33 h** | **STEREO** 24 k WAV | DERIVED — ⭐ **full-duplex training audio** |
| 7 | `training_data/stereo_dialogues_flac/` | FLAC mirror of #6 | 2.5 G | 1,367 flac | 38.3 h | FLAC | DERIVED — transport |
| 8 | `training_data/stereo_dialogues_flat/` | symlink index of #6 | 5.5 M | 1,367 symlinks | — | symlinks | DERIVED — for WhisperX |
| 9 | `training_data/word_timestamps/` (+`_flat`) | WhisperX large-v3 + wav2vec2 | 25 M | 1,367 json | — | json | DERIVED — timestamps |
| 10 | `training_data/tokenized_audio/` | Mimi RVQ tokens | 32 M | 1,362 npz `A,B` `(8,1255)` | — | npz | DERIVED — audio tokens |
| 11 | `training_data/tokenized_text/` | Urdu SPM tokens | 5.5 M | 1,362 npz `A,B` | — | npz | DERIVED — text tokens |
| 12 | `training_data/processed/train-001-of-001.parquet` | packed training set | 28 M | **1,362 rows** (`dialogue_id`,`A`,`B` `[9,T]`) | — | parquet | ⭐ DERIVED — **final training input** |
| 13 | `training_data/dataset_samples/` (+ root `dataset_samples/`) | demo samples | 23 M | `01_raw_cv`/`02_segments_merged`/`03_stitched_dialogues` | — | mp3/wav | demo only |
| 14 | `ur.txt` | Urdu web/news text (Nastaliq) | 5.8 G | ~26.65 M lines | — | text | ⭐ **SOURCE** — SPM tokenizer corpus |
| 15 | `urdu_alpaca.jsonl` | HF **Dolly-15K-ur** | 20 M | 15,015 | — | jsonl | SOURCE (LLM instruct; **not used** in Moshi training) |

Tokenizer artifacts (`weights/tokenizer/urdu_tokenizer.model/.vocab`, vocab **32,000**) are built from `ur.txt`.
Loose transport tarballs also exist in the source `training_data/` (`stereo_cv.tar`, `stereo_flac.tar`, `word_timestamps.tar`, `tok_audio.tar`) — dupes of derived dirs, ~16 G reclaimable.

---

## 2. Splits (train / val / test)
- **Common Voice (#1)** ships Mozilla's official splits (rows, header excluded): `train` 7,300 · `dev` 5,071 · `test` 5,072 · `validated` 67,845 · `other` 172,213 · `invalidated` 10,721.
- **Filtered manifest (#4)** = `train` + `validated` extras, with **dev/test speakers (`client_id`) and dev/test clip paths excluded** (leakage guard) → **train-only by construction**.
- **Derived training data (#5–#12): NO val/test split.** The parquet is literally `processed/train-…`; there is **no held-out val/test parquet or manifest**. Evaluation was done via **generated `samples/` continuations**, not a held-out set. → If you continue this work, **carve out a val/test split** for real evaluation.
- **TTS corpus (#2/#3):** no split (session folders only).

## 3. How each was used
```
Common Voice v25 (#1) ─ build_training_data.py ─► train_manifest.tsv (#4, 53.3h, dev/test excluded)
   ─ stitch_cv.py ─► stitched_cv (#5, mono, ORPHANED)
   ─ make_stereo_dialogues.py ─► stereo_dialogues_cv (#6, STEREO, 38.3h)  ── FLAC #7, flat #8
        ─ whisperx ─► word_timestamps (#9) ─ tokenize_text ─► tokenized_text (#11)
        ─ mimi ─► tokenized_audio (#10)
             ─ prepare_dataset ─► processed/train-*.parquet (#12, 1,362) ─► finetune
ur.txt (#14) ─ train_tokenizer ─► urdu_tokenizer.model/.vocab (32k)
urdu_alpaca.jsonl (#15): aux LLM data, NOT in the Moshi pipeline
TTS corpus (#2/#3): staged, NOT used
```

## 4. Full-duplex data note (#6)
Stereo **24 kHz 16-bit, L = speaker A / R = speaker B** — Moshi's full-duplex layout. But turns are
**non-overlapping**: when one speaker talks, the other channel is silence (verified from
`make_stereo_dialogues.py`). So it's the full-duplex *format* without real overlap/backchannel — see
`04_FULL_DUPLEX.md` for why that matters.

## 5. Source vs derived (what to preserve)
- **Preserve (not regenerable):** Common Voice corpus (#1), 10-hr TTS (#2), `ur.txt` (#14), the tokenizer, and the **model weights**.
- **Regenerable from `code/` (safe to drop to reclaim space):** #4–#13 and all the `*.tar` transport bundles (~30 G+ reclaimable).
