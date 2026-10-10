# Research Papers — Bibliography

The voice-model research drew on the papers below. **The core arXiv papers are now downloaded into
`research/papers/`** (they were not saved on the machine before — the project had only the *summaries*).
Papers without an open PDF are listed with their reference for retrieval.

## ✅ Downloaded (in `research/papers/`)
| File | Paper | Why it mattered |
|---|---|---|
| `2410.00037_Moshi.pdf` | **Moshi: a speech-text foundation model for real-time dialogue** — Kyutai, 2024 (arXiv:2410.00037) | The base model + Mimi codec + inner-monologue design this whole project fine-tunes. |
| `J-Moshi_2506.02979.pdf` | **Towards a Japanese Full-duplex Spoken Dialogue System (J-Moshi)** — Ohashi et al., Nagoya Univ., 2025 (arXiv:2506.02979) | Proves the `nu-dialogue/moshi-finetune` pipeline works for a non-English language; the closest precedent + the 69,000 h / 4,600 GPU-hr scale comparison. |
| `Kyutai_DSM_2509.08753.pdf` | **Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling (DSM)** — Kyutai, 2025 (arXiv:2509.08753) | The Moshi TTS "data factory" architecture (text-ahead delay) referenced for synthetic data generation. |
| `Full-Duplex-Bench_2503.04721.pdf` | **Full-Duplex-Bench: A Benchmark to Evaluate Full-Duplex Spoken Dialogue Models on Turn-taking** — 2025 (arXiv:2503.04721) | How full-duplex behaviour (turn-taking, backchannel, barge-in) is evaluated. |

## 📌 Referenced but no open PDF saved (retrieve as needed)
| Paper / source | Reference | Role in the research |
|---|---|---|
| **PersonaPlex** | NVIDIA, ICASSP 2026 | Highest-DMOS (3.90) full-duplex model; fine-tuned Moshi with **zero architecture changes**; source of the **negative-silence stitching** barge-in trick + role/persona conditioning. Find on arXiv/NVIDIA. |
| **Urdu TTS (Tacotron2 + voice cloning)** | MDPI 2024, id `2673-4591/87/1/112` | Best Urdu TTS baseline (MOS 4.2); the Urdu-TTS precedent. |
| **Urdu ASR (KICS/UET Lahore)** | Interspeech 2025 | Best Urdu ASR intelligibility (WER 13.1%), beating Meta MMS (25.6%). |
| **Fisher English corpus** | LDC | 7,303 two-speaker phone calls with **separate channels** — why channel-separated real data is prized (Kyutai/PersonaPlex both used it). |
| **WavLM** | Microsoft | The SSL model Mimi's semantic codebook is distilled from (why Mimi is English-centric for semantics). |
| **CosyVoice 2** | Alibaba | Candidate TTS for synthesizing the second speaker in synthetic dialogues. |
| **Qwen3-Omni** | Alibaba | The half-duplex bootstrap model in the PoC (understands Urdu out-of-the-box). |
| **Helium** | Kyutai | Moshi's 7B LLM backbone (what gets fine-tuned for Urdu). |
| **WhisperX** | github.com/m-bain/whisperX | Word-level forced alignment used in the pipeline (Stage 5). |

## Links found in the research notes
- Moshi: https://arxiv.org/abs/2410.00037 · https://kyutai.org/Moshi.pdf
- Kyutai DSM/TTS: https://arxiv.org/abs/2509.08753 · https://kyutai.org/tts · https://huggingface.co/kyutai/tts-1.6b-en_fr
- Full-Duplex-Bench: https://arxiv.org/abs/2503.04721
- J-Moshi: https://arxiv.org/abs/2506.02979 · https://nu-dialogue.github.io/j-moshi
- Fine-tune fork: https://github.com/nu-dialogue/moshi-finetune

> The synthesized reading of all these is in `research/urdu_fullduplex_research_summary.html` (the main
> summary), `moshi_finetune_urdu_guide.html`, and `moshi_tts_reference.html`.
