# 06 — Data-Pipeline Scripts (reference)

Every script is in `code/` in this handoff. Full-duplex stereo building is in `04_FULL_DUPLEX.md`;
training/finetune in `07`. This is the per-script reference for the data-prep side.

---

## `code/scripts/download_urdu_nashtiliq.py`
Downloads HF dataset `aaqibsaeed/databricks-dolly-15k-ur` (Urdu Nastaliq) → `urdu_alpaca.jsonl`
(`{"prompt","response"}` lines, 20 MB). Auxiliary text; the bulk tokenizer text is the pre-existing
`ur.txt` (6.1 GB, origin not scripted). Run: `python download_urdu_nashtiliq.py`.

## `code/scripts/train_tokenizer.py` + `verify_tokenizer.py`
SentencePiece **BPE** tokenizer.
- `input=ur.txt`, `model_prefix=urdu_tokenizer`, **`vocab_size=32000`** (must equal Moshi's English SPM size — init can't change vocab size), `model_type=bpe`, `character_coverage=0.9995`, `input_sentence_size=5_000_000`, `num_threads=8`.
- Special IDs matched to Moshi: `unk=0`, `bos=1`, `eos=2`, `pad=3` (line up with `text_padding_id=3`, `end_of_text_padding_id=0`, and retained init IDs `[0,3,32000]`).
- Out: `urdu_tokenizer.model` (875 KB), `urdu_tokenizer.vocab` (592 KB) → in handoff at `weights/tokenizer/`.
- `verify_tokenizer.py` encode/decode sanity-check on 7 Urdu + Roman/code-switch sentences.

## `code/scripts/build_training_data.py`
- Inputs: 10 hr TTS corpus (36 zips) + Common Voice v25 Urdu (`train/dev/test/validated.tsv`, `clip_durations.tsv`, `clips/`).
- (a) extracts TTS zips → `training_data/tts/<session>/` + `tts/manifest.tsv`; (b) symlinks CV clips → `training_data/common_voice/clips`; (c) builds `common_voice/train_manifest.tsv` = official `train` + `validated` extras, **excluding dev/test speakers (`client_id`) and dev/test clip paths** (leakage guard), with `duration_ms` joined. ~53 h / 80 speakers kept.
- ⚠️ The **TTS branch is built but unused downstream** (no TTS tokenization path). Run: `python build_training_data.py`.

## `code/scripts/stitch_cv.py`  (half-duplex / mono — orphaned)
Long **single-speaker** mono segments: `SR=24000`, soft cap 90 s / hard 120 s, 300 ms silence, 10 workers.
Groups CV by `client_id`, greedily packs clips with the ffmpeg **concat demuxer** → `stitched_cv/<client[:12]>/seg_XXXX.wav` + `.json` offsets + `manifest.tsv`. Log: **80 spk → 2,264 segs, 57.14 h**.
⚠️ **Not in the training path** — `make_stereo_dialogues.py` re-decodes raw MP3s instead, so `stitched_cv/` is an intermediate/demo artifact.

## `code/scripts/make_stereo_dialogues.py`  (full-duplex core)
→ see `04_FULL_DUPLEX.md`. L=A / R=B stereo, gender-aware pairing, 40 pairs → 1,367 dialogues, 38.33 h.

## `code/runpod_whisperx/process_all.py`  (word timestamps, GPU pod)
Per stereo dialogue → split L/R to mono → WhisperX `large-v3` (`--language ur`) → wav2vec2 forced
alignment (falls back to raw Whisper if no Urdu aligner) → tag words speaker A/B → merge+sort →
`word_timestamps/<pair>/dlg_XXXX.json` (`{speaker,word,start,end}`). Silent channels (amp < 0.01)
skipped; `--skip_existing` resumes. Format matches `tokenize_text.py`'s expectation.

## Framework tools (`code/moshi-finetune/tools/`)
| Tool | Role |
|---|---|
| `tokenize_audio.py` | Mimi RVQ tokenize stereo WAVs → `tokenized_audio/*.npz` `A,B` `[8,T]` (Mimi from `moshiko-pytorch-bf16`). Driven by `runpod_mimi/`. |
| `tokenize_text.py` | Urdu-SPM tokenize word JSONs → `tokenized_text/*.npz` `[T]` (pad 3, eot 0). `--text_tokenizer_path` is a fork patch. |
| `prepare_dataset.py` | Merge text+audio → `[9,T]` per speaker → `processed/train-001-of-001.parquet` (`dialogue_id`,`A`,`B`). Cross-checks name sets (drops 5 → 1,362). |
| `init_moshi_for_ft.py` | Init Moshi for Urdu (reinit text emb, add user stream `dep_q=16`). → `07`. |
| `decode_tokens.py` | Decode generated tokens → WAV + text using the Urdu SPM. |
| `zero_to_fp32.py` | Convert DeepSpeed Zero-3 checkpoint → fp32 safetensors. |

## Logs (in `results/logs/`)
`stitch_cv.log` (2,264 segs / 57.14 h), `stereo_dialogues.log` (1,367 dlg / 38.33 h).

## ⚠️ Missing `docs/`
The repo README links to `docs/architecture.md`, `docs/pipeline.md`, `docs/findings.md` — **none exist**.
The detailed findings/decoupling write-up referenced there is absent (partially reconstructed in `02`/`09`).
