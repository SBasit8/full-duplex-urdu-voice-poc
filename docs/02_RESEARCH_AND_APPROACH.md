# 02 — Research & Approach (why full-duplex, and the plan)

> Synthesized from the research notes now in `research/`. This is the "why" behind every script,
> dataset, and weight in this handoff. The single most important source is
> `research/urdu_fullduplex_research_summary.html`.

---

## 1. Goal
Build **the first natural Urdu full-duplex (simultaneous listen-and-speak) conversational voice AI**,
targeting **Pakistani Urdu**, by adapting Kyutai's **Moshi** speech-text model + its **Mimi** neural
audio codec.

The research frames this as filling a fourfold void: *"No full-duplex conversational system exists for
Urdu. No channel-separated Urdu conversation dataset exists anywhere. No production-quality open-source
Urdu TTS exists. No efficient Urdu LLM tokenizer designed for conversational use exists. Building this
system would produce the first of all four."*

**Two-track strategy:** use **Qwen3-Omni** immediately as a working (non-duplex) system, while building
the **Moshi adaptation in parallel** as the quality target. Whichever wins on naturalness + duplex
behaviour becomes production.

---

## 2. Full-duplex vs half-duplex (the core distinction)
| | Half-duplex (pipeline — the "old way") | **Full-duplex (Moshi — the priority)** |
|---|---|---|
| How | ASR → LLM → TTS, sequentially | One model listens + speaks **simultaneously, every 80 ms** |
| Latency | 500–900 ms | 160–200 ms |
| While speaking | deaf to the user | hears the user; can be interrupted |
| Turn-taking | a hard-coded rule | **emerges from training** |
| Examples | Siri, Google Assistant, ChatGPT Voice | Moshi, PersonaPlex |

Full-duplex is prioritized because Moshi processes **three streams at once** — agent text (the "Inner
Monologue"), agent audio, and user audio — with the two audio channels as **separated stereo**
(left = agent, right = user). Backchannels, barge-in, and natural turn-taking emerge from that design.
**Half-duplex still appears** only as the bootstrap PoC (Qwen3-Omni + Azure TTS) — a stepping stone,
explicitly "not full duplex... this is exactly what Moshi was built to solve."

---

## 3. How Moshi works (from `research/How Moshi predicts text.txt`)
- At **every 80 ms frame (12.5 Hz)** the model emits **9 tokens in parallel**: 1 text token (32k Urdu
  vocab, or padding) + 8 Mimi audio codes (codebook 0 = semantic; 1–7 = acoustic detail).
- Two stacked transformers: a **Temporal Transformer** (main, across time) + a **Depth Transformer**
  (predicts the 9 outputs, **text first**, then conditions the 8 audio codes on that text). This is what
  aligns what it "says" with what it "speaks."
- **Inner monologue:** the text stream is generated before audio — Helium's hidden "thinking layer."
- **Mimi codec:** 24 kHz audio → discrete tokens at 12.5 Hz, ~1.1 kbps, RVQ with 8 codebooks; codebook 0
  is semantic (WavLM-distilled). **Mimi is always frozen** — fine-tuning only touches Helium (+ Depth
  Transformer) + a new Urdu tokenizer + reinitialized text embeddings.

### Their own fine-tune post-mortem (important lesson)
Their run showed **text/audio misalignment**, diagnosed as a *data/training* problem, not architecture:
text supervision is very sparse (~5% of frames are real words, ~95% padding → text stream gets ~20× less
signal); text embeddings were reset from scratch; audio codebooks were retained (asymmetry); streams
decouple early. Recovery needs text-loss < ~3.0 over many epochs. Their run reached loss 3.75 at epoch 5
on ~6 GPU-hours — vs J-Moshi's ~4,600 GPU-hours. **Conclusion: architecture is sound; seeing real Urdu
words in the text stream at 0.13% of J-Moshi's compute is a positive signal — it needs far more data/compute.**

---

## 4. The Proof-of-Concept pipeline (`research/PROOF OF CONCEPT PIPLINE.txt`)
A deliberately half-duplex bridge to first prove Urdu speech-to-speech:
```
Mic (Urdu) → Qwen3-Omni Thinker (understands Urdu, incl. code-switching) → Urdu text
           → Azure TTS (AsadNeural, ur-PK) → WAV → Speaker
```
Qwen3-Omni understands Urdu out of the box (no training); its Talker can't *speak* Urdu, so Azure TTS
does the voice. Success here justifies the full Moshi build (synthetic data + real channel-separated
recordings).

---

## 5. Key findings / decisions
- **It's a DATA problem, not a compute problem.** Fine-tuning costs $100–$700; collecting 200 h of
  clean two-speaker Urdu costs $17k–$40k. Channel-separated Urdu conversation data doesn't exist — building it is the highest-ROI work.
- **Synthetic data teaches tasks; only real data gives naturalness** (backchannels, turn-taking). Even
  Kyutai needed ~2,000 h of real Fisher on top of 20,000 synthetic hours.
- **Precedents prove feasibility:** *PersonaPlex* (NVIDIA) fine-tuned Moshi with **zero architecture
  changes** to DMOS 3.90 using ~2,250 h synthetic + 1,217 h real Fisher, introducing the
  **negative-silence stitching trick** for barge-in. *J-Moshi* (Japanese) proves the `nu-dialogue/moshi-finetune`
  fork works for non-English and that **Mimi generalizes across languages without retraining**.
- **Tooling decision:** use the **`nu-dialogue/moshi-finetune`** fork (supports **custom text
  tokenizers** + `--init_text_embeddings` + `--no_whitespace_before_word`), NOT the official
  `kyutai-labs/moshi-finetune` (can't change the tokenizer → Urdu inner monologue would break).
- **Trained vs frozen:** Mimi encoder/decoder frozen; Helium (LoRA or full) + Depth Transformer + new
  Urdu SentencePiece tokenizer + reinitialized text embeddings are trained.

### DMOS-vs-data expectation (from the summary)
| Strategy | Expected DMOS | Timeline |
|---|---|---|
| Synthetic only | 2.3–2.8 | 4–6 mo |
| Synthetic + 50 h real stereo | 3.0–3.3 | 6–9 mo |
| Synthetic + 200 h real stereo | 3.3–3.7 | 12–18 mo |
| + custom Urdu TTS | 3.7–4.0 | 18–24 mo |

### 4-phase roadmap
1. **Mo 1–3:** PoC + Urdu tokenizer + confirm Mimi-Urdu compatibility.
2. **Mo 3–9:** 500–2,000 h synthetic + first Urdu Moshi fine-tune (DMOS ~2.5–3.0).
3. **Mo 6–18:** add 200 h real recorded stereo (DMOS 3.3–3.7).
4. **Mo 12–24:** custom Urdu multi-stream TTS + 5,000–20,000 h synthetic, production (DMOS ≥3.8).

---

## 6. Risks / open questions (from the notes)
- **Mimi's semantic codebook is English/WavLM-distilled** → Urdu retroflexes/aspirated stops/schwa may
  collapse to the same tokens. A real fix (fine-tuning Mimi on Urdu SSL) is a separate research project.
- **Real channel-separated Urdu data doesn't exist**; extracting it needs diarization (PyAnnote is weak
  on Urdu code-switching/overlap) + separation (bleed-through) + ASR (15–27% WER propagates errors).
- **Low-resource quality ceiling** (J-Moshi stayed well below English).
- **Nastaliq script** (no spaces, omitted short vowels) → tokenizer needs 2–3× tokens/word.
- **Code-switching** (Urdu-English) — no dataset, no TTS handles it natively.
- **Honorifics** (aap/tum/tu) — wrong register is offensive, not just unnatural.
- Open TODOs from `summary.txt`: write the full procedure in Notion; create the **Siglance GitHub repo**
  with code + README; investigate code-switching; resolve a **DeepSpeed ZeRO-3 / OOM** question; test
  whether zero-shot voice cloning can bootstrap the 20k-hour synthetic set.
- ⚠️ Note: an **RVQ codebook-count discrepancy** across notes (Moshi = 8 vs Kyutai TTS = 32) — 8 is
  Moshi's, 32 is the separate TTS model's.

---

## 7. Research files (in `research/`)
| File | What it is |
|---|---|
| `urdu_fullduplex_research_summary.html` | ⭐ The main 4-part research summary (full-duplex vs pipeline, Urdu challenges, data-separation problem + synthetic pipeline, base-model recommendation + roadmap + bibliography) |
| `moshi_finetune_urdu_guide.html` | Hands-on fine-tuning guide: what `moshi-finetune` does, official vs `nu-dialogue` fork, train-vs-frozen, dataset format/volume, 6-step command plan, `urdu_config.yaml` |
| `moshi_tts_reference.html` | Reference on Kyutai's Moshi **TTS** (the synthetic-data factory): architecture, DSM 1.28 s text-ahead, training phases, compute |
| `How Moshi predicts text.txt` | Per-frame joint text+audio prediction + their alignment post-mortem |
| `PROOF OF CONCEPT PIPLINE.txt` | The 3-step half-duplex PoC recipe |
| `summary.txt` | Working research/decision log + TODOs (HF tokens **redacted** — see `SECRETS/`) |

*(`text1.txt` was found alongside these but is **unrelated** — notes for a different cargo-email project — so it's excluded; see `SECRETS/README.md`.)*

**The actual papers** (Moshi, J-Moshi, Kyutai DSM, Full-Duplex-Bench) are downloaded into
`research/papers/`; the full reference list (incl. PersonaPlex, the Urdu TTS/ASR baselines, Fisher,
WavLM, etc.) is in `research/PAPERS_BIBLIOGRAPHY.md`. These were **not** on the machine originally — the
project had kept only the synthesized summaries above — so I fetched the open ones.
