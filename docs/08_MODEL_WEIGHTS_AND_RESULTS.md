# 08 — Model Weights & Results

All in `weights/` (real copies, not hardlinks).

---

## 1. The checkpoint — `weights/trained_models/moshi-urdu-step110/`
- `model.safetensors` — **3.97 GB**, dtype **fp16**, 475 tensors. Names include `depformer.layers.*.gating.*` → Moshi (Temporal + Depth transformer) confirmed.
- **No `config.json`/`params.json` in the dir.** The architecture kwargs live in the init model
  `moshiko-urdu-float32/moshi_lm_kwargs.json` (produced on the RunPod pod; **not stored locally** —
  regenerable via `run_init.sh`). You need that kwargs file to load the checkpoint.

> ⚠️ **Verify completeness.** The full uploaded model (**16 GB**) lives on HuggingFace at
> **https://huggingface.co/S-basit/moshi-urdu-poc** (private, ✅ confirmed present). The init float32
> model is **32 GB**. The training log says the model was saved "split in 2 checkpoint shards," but the
> local copy here is a **single 3.97 GB fp16 file with no `model.safetensors.index.json`** — i.e. a
> partial/single-shard export. **Pull the full 16 GB copy from the HF repo above** and treat that as
> authoritative.

## 2. Base & init (to load/reproduce)
- Base: `kyutai/moshiko-pytorch-bf16` (HF, public — re-downloadable).
- Init: `moshiko-urdu-float32` (~32 GB, reinit Urdu text embeddings + `dep_q=16` user stream) — made on the pod, regenerable via `code/runpod_finetune/run_init.sh`. Its `moshi_lm_kwargs.json` is required to instantiate the model.
- Tokenizer: `weights/tokenizer/urdu_tokenizer.model/.vocab` (SPM BPE, vocab 32,000).

## 3. Training result (`weights/trained_models/training2.log`)
- 5 epochs / **110 steps**, 2 GPUs, effective batch 64, max_length 2048 frames, DeepSpeed Zero-3, fp16, AdamW, LR ~3e-5.
- Final loss **≈5.39** (text **3.72**, audio **1.67**).
- Only `step_110` was saved (`--save_steps 200`).

## 4. Generated samples — `weights/trained_models/samples/`
- `0.wav … 9.wav` (10 continuations, ~1.9 MB each) + `0.txt … 9.txt`.
- The `.txt` show **real Urdu word tokens interleaved with `⁇` unknowns** — i.e. the model produces
  Urdu words but **not coherent sentences**. This matches the "decoupled/under-trained" diagnosis
  (text loss ~3.7); see `02_RESEARCH_AND_APPROACH.md` §3 and `09_STATUS_AND_MISSING.md`.
- `generate.log` (tiny), `training2.log` (143 K) also here. A `dryrun_sample.wav` is in `results/`.

## 5. Bundles — `code/bundles/`
`finetune_bundle.tar` (91 MB) and `mft.tar` (62 MB) are **snapshots of the `moshi-finetune/` training
repo** (code, incl. an embedded `.git`), **not weights or data**. The unpacked repo is at
`code/moshi-finetune/`. Kept for provenance.

## 6. What to do
- [ ] Pull the full **16 GB** model from HF `S-basit/moshi-urdu-poc` and reconcile vs the local 3.97 GB fp16 file.
- [ ] Save the **`moshi_lm_kwargs.json`** (init kwargs) alongside the weights — without it the checkpoint can't be loaded.
- [ ] For any real eval, generate against a **held-out set** (none exists yet — see `05` §2).
