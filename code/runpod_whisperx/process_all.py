i have #!/usr/bin/env python3
"""
process_all.py — Bulk word-level timestamping with WhisperX on GPU.

For each stereo dialogue WAV (L=Speaker A, R=Speaker B):
  1. Split into two mono channels
  2. Run WhisperX on each channel (Urdu, large-v3)
  3. Merge word-level timestamps into a single JSON per dialogue
  4. Output format matches moshi-finetune:
     [{"speaker":"A","word":"سلام","start":0.46,"end":0.82}, ...]

Usage:
  python process_all.py \
    --input_dir ./stereo_dialogues_cv \
    --output_dir ./word_timestamps \
    --model large-v3 \
    --language ur \
    --batch_size 16
"""

import argparse, json, os, sys, time, wave, tempfile
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Dir with stereo WAVs (recursive)")
    p.add_argument("--output_dir", required=True, help="Output dir for JSON files")
    p.add_argument("--model", default="large-v3", help="Whisper model size")
    p.add_argument("--language", default="ur", help="Language code")
    p.add_argument("--batch_size", type=int, default=16, help="WhisperX batch size")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--compute_type", default="float16", help="float16, int8, etc.")
    p.add_argument("--hf_token", default=None, help="HuggingFace token (for alignment model)")
    p.add_argument("--skip_existing", action="store_true", help="Skip if output JSON exists")
    return p.parse_args()

def split_stereo_to_mono(wav_path):
    """Split a stereo WAV into two mono numpy arrays (float32, normalized)."""
    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 2, f"Expected stereo, got {w.getnchannels()} channels"
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    left  = samples[:, 0].astype(np.float32) / 32768.0
    right = samples[:, 1].astype(np.float32) / 32768.0
    return left, right, sr

def save_mono_temp(audio_np, sr):
    """Save float32 mono audio to a temp WAV file (WhisperX needs a file path)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    int16 = (audio_np * 32767).astype(np.int16)
    with wave.open(tmp.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(int16.tobytes())
    return tmp.name

def transcribe_channel(model, audio_path, align_model, align_metadata, device, language, batch_size):
    """Run WhisperX transcription + alignment on one mono channel, return word list."""
    import whisperx

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    if not result.get("segments"):
        return []

    try:
        result = whisperx.align(
            result["segments"], align_model, align_metadata,
            audio, device, return_char_alignments=False
        )
    except Exception as e:
        print(f"    Alignment failed ({e}), using raw whisper timestamps")

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w and "word" in w:
                words.append({
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })
    return words

def process_one(wav_path, out_json, model, align_model, align_metadata, args):
    """Process one stereo dialogue WAV → output JSON."""
    left, right, sr = split_stereo_to_mono(wav_path)

    results = []
    for audio_np, speaker in [(left, "A"), (right, "B")]:
        if np.max(np.abs(audio_np)) < 0.01:
            continue
        tmp = save_mono_temp(audio_np, sr)
        try:
            words = transcribe_channel(
                model, tmp, align_model, align_metadata,
                args.device, args.language, args.batch_size
            )
            for w in words:
                w["speaker"] = speaker
            results.extend(words)
        finally:
            os.unlink(tmp)

    results.sort(key=lambda w: w["start"])
    reordered = [{"speaker": w["speaker"], "word": w["word"],
                  "start": w["start"], "end": w["end"]} for w in results]

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(reordered, f, ensure_ascii=False, indent=2)
    return len(reordered)

def main():
    args = parse_args()
    import whisperx

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    wavs = sorted(input_dir.rglob("dlg_*.wav"))
    print(f"Found {len(wavs)} dialogue WAVs in {input_dir}")

    if args.skip_existing:
        before = len(wavs)
        wavs = [w for w in wavs if not (output_dir / w.relative_to(input_dir).with_suffix(".json")).exists()]
        print(f"Skipping {before - len(wavs)} already processed, {len(wavs)} remaining")

    if not wavs:
        print("Nothing to process."); return

    # Load model once
    print(f"Loading WhisperX model ({args.model}) on {args.device}...")
    model = whisperx.load_model(args.model, args.device, compute_type=args.compute_type,
                                language=args.language)

    # Load alignment model
    print(f"Loading alignment model for '{args.language}'...")
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=args.language, device=args.device,
            model_name=None
        )
        print("Alignment model loaded OK")
    except Exception as e:
        print(f"No alignment model for '{args.language}': {e}")
        print("Will use raw Whisper word timestamps (still good)")
        align_model, align_metadata = None, None

    # Process all
    t0 = time.time()
    total_words = 0
    for i, wav_path in enumerate(wavs, 1):
        rel = wav_path.relative_to(input_dir)
        out_json = output_dir / rel.with_suffix(".json")

        try:
            n = process_one(wav_path, str(out_json), model, align_model, align_metadata, args)
            total_words += n
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (len(wavs) - i)
            print(f"  [{i}/{len(wavs)}] {rel}  words={n}  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")
        except Exception as e:
            print(f"  [{i}/{len(wavs)}] ERROR {rel}: {e}")

    elapsed = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"Processed: {len(wavs)} dialogues, {total_words} words, {elapsed/60:.1f} min")
    print(f"Output: {output_dir}")

if __name__ == "__main__":
    main()
