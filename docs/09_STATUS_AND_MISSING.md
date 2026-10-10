# 09 — Status: Done / In-Progress / Planned + Missing Resources

A **proof-of-concept**, not a product. It proved the `nu-dialogue/moshi-finetune` pipeline runs
end-to-end for Urdu and that Moshi/Mimi accept Urdu — but the model is under-trained (38 h vs the
thousands of hours needed).

---

## ✅ DONE
- **Full data pipeline** built and run end-to-end (tokenizer → corpus → stereo dialogues → whisperx timestamps → mimi+text tokenization → parquet → init → finetune → generate → decode). See `03`.
- **Urdu SentencePiece BPE tokenizer** (32k vocab) trained from `ur.txt`.
- **Synthetic full-duplex dataset**: 1,367 stereo dialogues / **38.33 h** (L=agent, R=user) from Common Voice v25 Urdu; 1,362 made it into the training parquet.
- **Dual-stream full-duplex init** (`dep_q=16`, user-stream heads, reinit Urdu text embeddings).
- **Fine-tune completed**: 5 epochs / 110 steps on 2 GPUs; checkpoint `moshi-urdu-step110` + generated samples + logs. Uploaded to private HF `S-basit/moshi-urdu-poc`.
- **RunPod job scripts** for the 3 GPU stages (whisperx, mimi, finetune), each with `INSTRUCTIONS.md`.
- **Research**: full-duplex literature review, Moshi/Mimi study, PoC recipe, roadmap (`02` + `research/`).

## ⏸️ PAUSED AFTER THE POC (confirmed)
**Nothing was executed after the POC** — it's paused here by design, pending data. To resume we need far
more **2-channel stereo conversational Urdu data** (real-recorded or TTS-synthesized), then train at
scale (30+ epochs). See `12_WORK_SUMMARY_worked-didnt-pivots.md` → "Direction after the POC". Leftover
partials from the POC itself:
- **Model quality** — emits real Urdu words but no coherent sentences (text loss ~3.7, "decoupled" regime).
- **Half-duplex PoC (Qwen3-Omni + Azure TTS)** — designed in the research notes only, not built.
- **`stitch_cv.py` mono segments** (57 h) — produced but **orphaned** (not in the training path).

## 🗓️ PLANNED / NOT BUILT (from README "Next steps" + research)
- Train **15–30 epochs** to resolve text↔audio decoupling; raise text loss weight; init from pre-trained Urdu embeddings; **LoRA on the text path only** (not implemented — needs framework changes).
- Acquire **200+ h real conversational Urdu** (channel-separated) — the highest-ROI item.
- **Overlap/backchannel** augmentation of the stitcher (PersonaPlex negative-silence trick) for true full-duplex timing.
- Wire up the **TTS corpus** (staged, unused) and a synthetic-data generator (GPT-4o/Qwen scripts → per-speaker TTS → stereo stitch).
- Custom Urdu multi-stream TTS; scale synthetic to 5,000–20,000 h (roadmap Phase 4).
- Resolve the **DeepSpeed Zero-3 / OOM** question noted in `summary.txt`.
- Publish to the planned **Siglance GitHub repo** + Notion write-up.

## 🐞 KNOWN ISSUES / LIMITATIONS
- **Synthetic dialogues are turn-disjoint** (no overlap/backchannel, no semantic coherence) — a core reason full-duplex behaviour doesn't emerge. (`04`)
- **Mimi's semantic codebook is English/WavLM-distilled** — Urdu-specific phonemes may collapse; a real fix (Urdu Mimi) is a separate project.
- **README numbers vs log mismatch** — README says H100 / 5.3 h / ~105 steps; `training2.log` shows 2 GPUs / ~35 min / 110 steps. Reconcile before quoting.
- **5 dialogues silently dropped** (1,367 → 1,362) by the text/audio set-mismatch guard; not tracked.
- **Fork patches not upstreamed** — re-cloning `moshi-finetune` loses them (`07` §6).
- **Loose transfer tarballs** in `training_data/` (`stereo_cv.tar`, `stereo_flac.tar`, `tok_audio.tar`, `word_timestamps.tar`) + root (`finetune_bundle.tar`, `mft.tar`).

---

## 📦 MISSING / STORED ELSEWHERE (grab for the record)
| Item | Where / status |
|---|---|
| **`docs/` directory** | README links `docs/architecture.md`, `docs/pipeline.md`, `docs/findings.md` — **they don't exist**. The detailed findings/decoupling write-up is missing (partly reconstructed in `02`/`09`). Recreate them. |
| **HF model repo** | **https://huggingface.co/S-basit/moshi-urdu-poc** — private, ~16 GB full checkpoint, ✅ confirmed present. Local copy (`weights/trained_models/`) is a 3.97 GB fp16 partial — pull the HF copy as authoritative + back it up. |
| **`ur.txt` provenance** | 6.1 GB Urdu text corpus, origin not scripted — record where it came from. |
| **Init model** `moshiko-urdu-float32` (~30 GB) | Produced on the RunPod pod; not stored locally. Regenerable via `run_init.sh` from the base model. |
| **Base model** `kyutai/moshiko-pytorch-bf16` | HuggingFace (public) — re-downloadable. |
| **RunPod pods** | Ephemeral (terminated). Only the `runpod_*` scripts remain — the pods themselves are gone. |
| **Exposed HF tokens** | In `~/summary.txt` — **rotate** (see `SECRETS/README.md`). |
| **`text1.txt`** | Unrelated (different project) — excluded; file it elsewhere. |
| Conda envs `urdu-moshi`, `whisperx-test` | Not copied (13 GB) — package lists in `env/`, recreate via `10`. |
