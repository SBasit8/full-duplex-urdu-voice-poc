# 12 — Overall Work Summary: What Worked, What Didn't, and Why We Shifted

The single consolidated read on the voice-model work. Sources: the repo README (author's own
findings), the research notes (`research/`), and the code/data analysis. This is the "if you read one
doc" summary.

---

## TL;DR
We adapted Kyutai's **Moshi** (full-duplex speech model) to **Urdu**: new Urdu tokenizer, synthesized
stereo two-speaker dialogues from Common Voice, tokenized with Mimi, and fine-tuned Moshi's LM (Mimi
frozen). The **pipeline works end-to-end and the architecture transfers** — but with only **38 hours**
of synthetic data and **~6 GPU-hours**, the model produces Urdu words without coherent sentences. It's a
**validated proof-of-concept, not a usable model**. The bottleneck is **data + compute, not architecture**.

---

## ✅ What WORKED
- **Pipeline integrity** — raw Common Voice → tokenizer → stereo dialogues → timestamps → Mimi/text tokens → parquet → fine-tune → generated samples, end-to-end, **with zero architectural changes** to Moshi.
- **Tokenizer swap** — a 32k Urdu SentencePiece cleanly replaced Moshi's 32k English SPM.
- **Text-embedding learning** — real Urdu words emerged in the text stream (e.g. پنجاب, خوشی, زندگی, گوادر, موٹرسائیکل).
- **Audio output** — decoded to a speech-like signal (not noise).
- **Dual-stream full-duplex** — L/R (agent/user) channels preserved; both streams generated independently (16 audio heads = 8 Moshi + 8 user).
- **Mimi is language-agnostic** — the frozen codec handled Urdu audio without retraining (matches J-Moshi's finding).
- **Cheap to run** — whole POC ≈ **$17–22**; a clean rerun ≈ $5–7.

## ⚠️ What DIDN'T (was rough)
- **Audio coherence** — speech-like but not fluent Urdu.
- **Text coherence** — real words, no sentence structure.
- **Text↔audio alignment** — the two streams don't say the same thing yet ("**decoupling**").
- **Voice consistency** — voice drifts within a single 20-second generation.

## 🔍 WHY (root causes — all data/training, not architecture)
- **38 h vs ~69,000 h.** This POC used ~0.02–0.05% of J-Moshi's data/compute (6 GPU-hours vs ~4,600). Urdu words emerging *at all* at that scale is a positive signal.
- **The decoupling phenomenon (recoverable).** Moshi's Depth Transformer predicts text first, then conditions audio on it. Early in training the text is near-random, so the audio side learns to *ignore* it → the streams decouple. With enough training (text loss < ~3.0; this run stalled at ~3.7) they re-couple. We're stuck in the decoupled phase purely from compute budget.
- **Sparse text supervision** — only ~5% of frames are real words (~95% padding), so the text stream gets ~20× less signal; embeddings were also reinit-from-scratch. (See `02_RESEARCH_AND_APPROACH.md` §3.)
- **Turn-disjoint synthetic data** — the stereo dialogues never overlap (one channel silent per turn) and have random, unrelated turn content, so there's no real backchannel/barge-in or semantic coherence to learn. (See `04_FULL_DUPLEX.md`.)

---

## 🔀 Decisions & PIVOTS (what we shifted away from, and why)
| We shifted… | From → To | Why |
|---|---|---|
| **Fine-tuning framework** | official `kyutai-labs/moshi-finetune` → **`nu-dialogue/moshi-finetune`** (patched fork) | The official repo **can't change the text tokenizer** → the Urdu inner-monologue would be broken. The J-Moshi fork supports a custom SPM (`--text_tokenizer_path`) + `--init_text_embeddings`. |
| **Checkpoint saving** | `accelerator.save_state` (~70 GB optimizer states) → `save_model` (~16 GB weights only) | 70 GB didn't fit the disk budget; we only needed the weights. |
| **TTS corpus (10 hr)** | staged → **dropped from the training path** | Nothing consumed it; the audio pipeline used only Common Voice-derived stereo. |
| **`stitch_cv.py` mono segments** | intermediate → **orphaned** | `make_stereo_dialogues.py` re-decodes raw MP3s directly, so the 57 h of mono segments aren't used in training. |
| **`urdu_alpaca.jsonl` (Dolly-15k-ur)** | explored → **unused** | LLM instruction text, not part of the speech pipeline. |

---

## ➡️ Direction after the POC (confirmed)
**Nothing was executed after the POC — the work is paused here, by design, pending data.** The POC did
its job (validated the architecture end-to-end); it was never going to produce a usable model at 38 h.
The blocker and the required next move are clear:

- **We need far more voice + conversational data than exists.** Two ways to get it: (a) **collect real**
  two-speaker Urdu conversations, and/or (b) **create it synthetically** with real/voice-cloning **TTS
  models** — generate large volumes of Urdu dialogue audio, then train on that.
- **It must be 2-way STEREO, not single-channel.** Full-duplex means the model listens and speaks at the
  same time, so each speaker has to live on its **own channel** (L = agent, R = user). Mono /
  single-channel data **cannot** teach simultaneous listen+speak, turn-taking, or barge-in. So whatever
  we collect or synthesize has to be **channel-separated stereo** — and ideally with real overlap /
  backchannel, unlike the turn-disjoint (one-channel-silent) data this POC used.
- **Then** fine-tune Moshi on that larger 2-channel dataset for many more epochs (30+) — which is what
  resolves the text↔audio decoupling and yields demoable Urdu full-duplex.

**In short:** POC proved it works → **paused** → resume once we have enough **2-channel stereo
conversational Urdu data** (real-recorded or TTS-generated), then train at scale.

---

## 🎯 The "last working full-duplex" run — exact artifacts (all in this handoff)
This is the complete set that constitutes the last (and only) working full-duplex run:

| Piece | Path in handoff |
|---|---|
| Full-duplex data builder | `code/scripts/make_stereo_dialogues.py` |
| Training + inference code | `code/moshi-finetune/` (`finetune.py`, `generate.py`, `tools/`), `code/runpod_finetune/` |
| Full-duplex **dataset** | `datasets/training_data/stereo_dialogues_cv/` (1,367 dlg, 38.3 h) |
| Exact **training input** | `datasets/training_data/processed/train-001-of-001.parquet` (1,362 rows) |
| Tokenizer | `weights/tokenizer/urdu_tokenizer.model/.vocab` |
| **Checkpoint** (last = only) | `weights/trained_models/moshi-urdu-step110/model.safetensors` |
| Generated **samples** + logs | `weights/trained_models/samples/`, `training2.log`, `generate.log` |

### ⚠️ To actually LOAD/RUN it you still need two things (not on disk):
1. **The full 16 GB weights** — live on HuggingFace at **https://huggingface.co/S-basit/moshi-urdu-poc** (private, ✅ confirmed present). The local copy here is 3.97 GB fp16 (a partial/single-shard export), so pull the full model from that repo.
2. **`moshi_lm_kwargs.json`** (the model's init kwargs) — it lived on the deleted RunPod pod and is **not on this machine**. Get it from the HF repo above, or regenerate via `code/runpod_finetune/run_init.sh`. Without it the checkpoint can't be instantiated.

**Reproduce/run:** `07_TRAINING_AND_FINETUNE.md` §5 + `code/runpod_finetune/INSTRUCTIONS.md`.

---

## 📈 Next steps (from the README, cheapest → highest-impact)
| Step | Cost | Gain |
|---|---|---|
| Train 15–30 epochs on the same data | $30–60 | resolves decoupling; better text |
| Higher text-loss weight | ~$30 | faster vocab convergence |
| Init from pre-trained Urdu embeddings | ~$15 | skip ~10 epochs of vocab learning |
| LoRA on text path only (not yet implemented) | ~$10/iter | cheap tokenizer iteration |
| **Acquire 200+ h real conversational Urdu** | major data work | fundamentally better quality |
| **Full retrain on real data, 30+ epochs** | $500–2,000 | demoable Urdu Moshi |
