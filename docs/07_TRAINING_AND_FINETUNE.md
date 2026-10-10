# 07 — Training & Fine-Tune

Framework: **`nu-dialogue/moshi-finetune`** (patched fork, vendored at `code/moshi-finetune/`) — chosen
over the official `kyutai-labs/moshi-finetune` because it supports a **custom text tokenizer** (needed
for the Urdu inner monologue). accelerate + **DeepSpeed Zero-3**, FP16 compute / float32 master weights,
**full fine-tune (not LoRA)**.

---

## 1. Base model & init
- **Base:** `kyutai/moshiko-pytorch-bf16` (Temporal ~7 B + Depth ~700 M + Mimi codec).
- **Init** (`tools/init_moshi_for_ft.py`, via `runpod_finetune/run_init.sh`):
  - `--init_text_embeddings` → re-sample `text_emb` + `depformer_text_emb` from a Gaussian fit to the original stats; **retain original embeddings only for IDs `[0, 3, 32000]`**.
  - `--extend_modules_for_user_stream` → `dep_q=16`, `depformer_context=16` (8 Moshi + 8 user codebooks) + user-stream heads → **dual-stream full-duplex**.
  - Saves float32 → `init_models/moshiko-urdu-float32/` (`model.safetensors` ~30 GB + `moshi_lm_kwargs.json`).

## 2. Dataset format the trainer consumes
Parquet (`training_data/processed/train-001-of-001.parquet`, 28 MB) — columns `dialogue_id`, `A`, `B`;
each of `A`/`B` is a **`[9, T]`** int32 matrix (row 0 = text tokens, rows 1–8 = Mimi RVQ audio), T at
12.5 Hz. ≤100k dialogues per parquet. **1,362** dialogues.

## 3. Hyperparameters (`run_finetune.sh` + `finetune.py`, cross-checked vs `training2.log`)
| Param | Value |
|---|---|
| LR (temporal & depth) | `3e-5`, linear warmup `--num_warmup_steps 100` |
| Batch | `PER_DEVICE_BATCH=2`, `GRAD_ACCUM=16`, 2 GPUs → **effective 64** (README default is 4/8; run used 2/16) |
| Epochs / steps | 5 epochs → **110 optimizer steps** |
| Seq length | `--max_length 2048` frames (~164 s @12.5 Hz), `--min_length 128` |
| Optim | AdamW betas (0.9, 0.95), eps 1e-5, `weight_decay 0.1`, `--activation_checkpointing` |
| Save | `--save_steps 200` → only `step_110` saved |
| Loss weights | `text_padding_loss_weight=0.5`, `semantic_loss_weight=100.0`, `acoustic_loss_weight=1.0` |
| Misc | `NO_TORCH_COMPILE=1` (avoids torch.compile/DeepSpeed conflict) |

**Loss** = text (non-pad + 0.5×pad) + audio (semantic upweighted 100× vs acoustic), summed over Moshi +
user streams. Final (`training2.log`): total ≈ **5.4** (text ≈ 3.7, audio ≈ 1.65).

**DeepSpeed:** `code/moshi-finetune/ds_configs/zero3-fp16-warmlr-act_ckpt.json` — Zero-3, fp16
(`initial_scale_power:10`), AdamW, WarmupLR, `stage3_gather_16bit_weights_on_model_save:true` (fork patch),
partitioned activation checkpointing.

## 4. RunPod jobs (`code/runpod_*`, each has `INSTRUCTIONS.md`)
Pattern: rent pod → `croc`/`scp` code+data in → `setup.sh` → run → `croc` results out → terminate.

| Job | GPU / image | Does | ~Cost |
|---|---|---|---|
| `runpod_whisperx/` | A10G 24 GB · `pytorch:2.1.0-cu118` | word timestamps (WhisperX large-v3, ur) | ~$0.40 |
| `runpod_mimi/` | A10G 24 GB · `pytorch:2.4.0-cu121` | Mimi audio tokenization (`moshi==0.1.0`) | ~$0.05 |
| `runpod_finetune/` | **2× A100 80 GB**, 100 GB vol · `pytorch:2.4.0-cu121` | init + finetune + generate | ~$15–20 |

`runpod_finetune/` setup pins: `moshi==0.1.0`, `accelerate==1.1.1`, `datasets==3.1.0`,
`deepspeed==0.15.4`, `transformers==4.46.1`, `sentencepiece`, wandb, soundfile.
Scripts: `run_init.sh`, `run_finetune.sh`, `run_generate.sh` / `run_generate_few.sh`,
`merge_shards.py` (consolidate sharded safetensors), `test_my_voice.py` (encode your own WAV as speaker
A → 1-row parquet for custom-prompt inference).

## 5. Checkpoint → inference → decode
1. `python -m tools.zero_to_fp32 <step_dir> <step_dir>_fp32 --moshi_lm_kwargs_path .../moshiko-urdu-float32/moshi_lm_kwargs.json` (+ `merge_shards.py`).
2. `generate.py` (via `run_generate*.sh`): prompt 125 frames (10 s), generate 250 frames (20 s), temp 0.8.
3. `python -m tools.decode_tokens --tokens_dir … --text_tokenizer_path …/urdu_tokenizer.model` → `samples/{0..9}.wav`+`.txt`.

Local outputs: `weights/trained_models/{moshi-urdu-step110, samples, training2.log, generate.log}`.
Uploaded to private HF **`S-basit/moshi-urdu-poc`** (~16 GB).

## 6. ⚠️ Fork patches (not upstreamed — live only in `code/moshi-finetune/`)
README lists ~8 patches on top of `nu-dialogue/moshi-finetune` (e.g. `--text_tokenizer_path`,
`np.concat`→`np.concatenate`, `save_model` instead of `save_state`, decode-transcript enable,
`stage3_gather_16bit_weights_on_model_save`). **Re-cloning the upstream repo loses these** — keep this
vendored copy. See the README (`_MOSHI_URDU_README.md`) "Patches" section.

## 7. Reproduce (high level)
```
# 0-4 local (CPU): tokenizer + corpus + stereo dialogues
python train_tokenizer.py && python build_training_data.py && python make_stereo_dialogues.py
# 5 pod: whisperx word timestamps ; 6 pod: mimi tokenize ; 7 local: tokenize_text ; 8 local: prepare_dataset
# 9-10 pod (2×A100): run_init.sh then run_finetune.sh
# 11 pod: run_generate.sh -> decode_tokens
```
Environments in `10_ENVIRONMENT_AND_SETUP.md`.
