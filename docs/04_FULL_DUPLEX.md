# 04 — Full-Duplex (the most important part)

Full-duplex = the model **listens and speaks at the same time** (every 80 ms), with the two audio
streams carried as **separated stereo**: **left channel = agent (speaker A), right channel = user
(speaker B)**. This is the format Moshi requires and what this whole project is oriented around.

Two things make this POC full-duplex: **(1)** the training data is built as stereo two-speaker
dialogues, and **(2)** the model is extended with a user-audio stream at init. Both are covered here.

---

## 1. Building the full-duplex data — `code/scripts/make_stereo_dialogues.py`
Because Common Voice is single-utterance read speech (no conversations), dialogues are **synthesized**
by interleaving turns from two different speakers into a stereo file.

**Config (top of file):** `SR=24000`, `GAP_MS=300` (silence between turns), `DLG_SOFT_MS=100_000`
(start a new dialogue past 100 s), `DLG_CAP_MS=120_000` (hard 120 s cap), `N_WORKERS=8`, `seed=42`.
Paths: in = `training_data/common_voice/train_manifest.tsv` + `clips/`; out = `training_data/stereo_dialogues_cv/`.

**How it works:**
1. **Group + gender** — group CV clips by `client_id`, record gender (male/female/unknown), drop zero-length clips.
2. **Gender-aware pairing (one pair per speaker)** — pair female↔male as **MF** pairs; remaining males + all unknowns paired two-at-a-time as **XX** pairs (odd leftover dropped). Per log: **40 pairs (14 MF + 26 XX)**. Pair name = `{a_id[:8]}_{b_id[:8]}_{tag}`.
3. **Decode** — `ffmpeg` decodes each source MP3 to 24 kHz **mono** int16 PCM (decodes raw CV MP3s directly; does **not** reuse the Stage-3 stitched segments).
4. **Build dialogues (`build_pair`)** — clips sorted then deterministically shuffled (seeded by `hash((a_id,b_id))`); pack **alternating turns starting with A**. Each turn appends `[gap, clip]` to the active speaker's channel and `[gap, equal-length silence]` to the other — so **exactly one channel has audio per turn**, and both channels stay sample-aligned and equal length. Soft-cap 100 s, hard-cap 120 s. Metadata per turn `{spk, path, start_s, end_s, sentence}`.
5. **Write** — combine to interleaved stereo (`L=[0::2]`, `R=[1::2]`) via Python `wave` (`nchannels=2, sampwidth=2, 24000`) → `dlg_XXXX.wav` + sibling `dlg_XXXX.json` (`{pair, speaker_A, speaker_B, duration_s, num_turns, turns[...]}`). Multiple dialogues per pair until clips run out.
6. **Manifest** — `stereo_dialogues_cv/manifest.tsv`: `dialogue_path, speaker_A, speaker_B, duration_s, num_turns, transcript` (transcript is turn-tagged `"[A] … [B] …"`).

**Result (`stereo_dialogues.log`):** **1,367 dialogues, 38.33 hours** — the actual training audio.

---

## 2. Making the model full-duplex — dual-stream init
`code/moshi-finetune/tools/init_moshi_for_ft.py` (run via `runpod_finetune/run_init.sh`):
- `--extend_modules_for_user_stream` sets **`dep_q=16`** and `depformer_context=16` (**8 Moshi codebooks + 8 user codebooks**) and adds the depth-transformer **user-stream heads** — so the model predicts **both its own audio and the user's audio**. This is the architectural switch that makes training full-duplex.
- `--init_text_embeddings` re-samples the text embeddings for the new Urdu vocab (retaining original IDs `[0, 3, 32000]`).

At training time (`prepare_dataset.py`) each speaker becomes a `[9, T]` matrix (1 text + 8 audio); `A`
and `B` are the two streams the model learns to co-predict.

---

## 3. ⚠️ Key limitation (read this)
The synthesized dialogues are **turn-disjoint**: when one speaker talks, the other channel is **pure
silence** — there is **no true overlap, no backchannel, no barge-in**. It teaches turn *structure* but
not real full-duplex *timing*. Turn content is also random (unrelated clips), so there's **no semantic
dialogue coherence** to learn. This is a primary reason the model doesn't learn natural conversational
behaviour, and it's exactly what the research says real (or overlap-augmented) data is needed to fix —
e.g. PersonaPlex's **negative-silence stitching** trick for barge-in (see `02_RESEARCH_AND_APPROACH.md`).

**To improve full-duplex behaviour next:** introduce real channel-separated recordings, and/or augment
the stitcher with overlapping turns + backchannels (negative gaps) instead of strictly alternating silence.

---

## 4. Data & code locations
| Thing | Path in handoff |
|---|---|
| Stereo dialogue builder | `code/scripts/make_stereo_dialogues.py` |
| Dual-stream init | `code/moshi-finetune/tools/init_moshi_for_ft.py`, `code/runpod_finetune/run_init.sh` |
| Full-duplex data | `datasets/training_data/stereo_dialogues_cv/` (+ `_flat`, `_flac`) |
| Word timestamps | `datasets/training_data/word_timestamps/` |
| Final training parquet | `datasets/training_data/processed/train-001-of-001.parquet` |
| Build log | `results/logs/stereo_dialogues.log` |
