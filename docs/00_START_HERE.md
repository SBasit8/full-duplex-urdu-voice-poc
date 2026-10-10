# 📁 Urdu Moshi — Voice Models Handoff (code + datasets + weights + research)

**Complete, self-contained handoff for the Urdu full-duplex voice-model work.** This folder
**physically contains** the code, model weights, datasets, demo media, research, and env specs as
**genuine independent file copies** (no hardlinks, no symlinks) — safe to zip and upload to a shared
drive for the dev team. Redundant duplicates (`.tar` bundles that copied dirs already present) and
break-on-move symlink indexes were stripped; see `datasets/MANIFEST_EXCLUDED.md`.

Compiled **2026-08-27**. Source: `/home/ubuntu/vs_code/moshi_urdu ` (note the trailing space).

---

## What this is
Fine-tuning Kyutai **Moshi** (+ **Mimi** codec) into an **Urdu full-duplex** conversational voice model.
A proof-of-concept: the full data→training→inference pipeline runs, the model speaks Urdu words, but
it's under-trained (38 h vs thousands needed). Full-duplex (simultaneous listen+speak via separated
stereo streams) is the priority; a half-duplex Qwen3-Omni+Azure PoC is the bootstrap.

## Read in this order
| # | Doc | What |
|---|---|---|
| 00 | `00_START_HERE.md` | This index |
| 01 | `01_PROJECT_OVERVIEW.md` | The idea, key numbers, first things to check |
| ⭐ | **`12_WORK_SUMMARY_worked-didnt-pivots.md`** | **The overall summary — what worked, what didn't, why we shifted, and the exact "last working full-duplex" artifacts. Read this first if you read nothing else.** |
| 02 | `02_RESEARCH_AND_APPROACH.md` | Why full-duplex; Moshi/Mimi; PoC; findings; roadmap |
| 03 | `03_ARCHITECTURE_AND_PIPELINE.md` | The 11-stage pipeline + dependency graph |
| 04 | `04_FULL_DUPLEX.md` | ⭐ Stereo dialogues + dual-stream init (the priority) |
| 05 | `05_DATASETS.md` | Inventory, sizes, hours, **splits**, how each was used |
| 06 | `06_DATA_PIPELINE_SCRIPTS.md` | Each data-prep script |
| 07 | `07_TRAINING_AND_FINETUNE.md` | moshi-finetune config, RunPod, hyperparams |
| 08 | `08_MODEL_WEIGHTS_AND_RESULTS.md` | Checkpoint, samples, the fp16-vs-16GB caveat |
| 09 | `09_STATUS_AND_MISSING.md` | Done / in-progress / planned + missing resources |
| 10 | `10_ENVIRONMENT_AND_SETUP.md` | Recreate the conda envs from `env/` |

## The gathered material (folders)
| Folder | Contents | Copy type |
|---|---|---|
| `code/` | all scripts, `moshi-finetune/` repo, `runpod_*` jobs, `.tar` bundles | **real** |
| `weights/` | `trained_models/` checkpoint + samples + logs; `tokenizer/` | **real** |
| `datasets/` | Common Voice v25, `ur.txt`, `training_data/` (stereo dialogues WAV+FLAC, stitched, tokenized, parquet, timestamps), 10-hr TTS, samples | **real copies** |
| `research/` | full-duplex summary, guides, notes (HF tokens redacted) | **real** |
| `demo_media/` | moshi audio/video + SpokenWOZ sample | **real** |
| `env/` | conda + pip lists for `urdu-moshi` and `whisperx-test` | exports |
| `results/` | logs + `dryrun_sample.wav` | **real** |
| `SECRETS/` | 🔒 note on exposed HF tokens to rotate | — |
| `_MOSHI_URDU_README.md` | the original repo README (verbatim) | **real** |

## ⚡ 60-second summary
- **Pipeline (all in `code/`):** `ur.txt`→SPM tokenizer; Common Voice v25→filtered manifest→`stitch_cv`→**`make_stereo_dialogues`** (L=A/R=B stereo)→WhisperX word timestamps→Mimi+SPM tokenize→parquet→**`finetune.py`** (DeepSpeed Zero-3)→`step110`→generate/decode.
- **Data:** 294 h CV → 38.33 h stereo dialogues (1,367) → 1,362 training rows. **No val/test split** (eval was via generated samples).
- **Weights:** `moshi-urdu-step110` (local 3.97 GB fp16; full 16 GB on HF `S-basit/moshi-urdu-poc`).
- **Result:** real Urdu words, no coherent sentences — under-trained, architecture sound.

## 🚨 Do first
1. **Reconcile the checkpoint** (local fp16 looks partial vs 16 GB HF) + keep the init `moshi_lm_kwargs.json` (needed to load it). (`08`)
2. **Rotate the 2 HuggingFace tokens** exposed in `summary.txt`. (`SECRETS/`)
3. **Recreate the missing `docs/`** the README links to. (`09`)
4. Datasets are **real copies** now — just zip and upload; nothing to materialize. Excluded/regenerable items are listed in `datasets/MANIFEST_EXCLUDED.md`.

## ⚠️ Notes
- Full-duplex data is stereo but **turn-disjoint** (no real overlap/backchannel) — a key reason conversational timing didn't emerge; fix with real data or overlap augmentation. (`04`)
- The `moshi-finetune/` fork has **un-upstreamed patches** — don't re-clone upstream and lose them. (`07`)
- `text1.txt` found near the research notes is a **different project** — excluded.
