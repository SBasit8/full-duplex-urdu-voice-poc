"""
test_my_voice.py — End-to-end inference test with your own voice.

Pipeline:
  1. Loads your WAV file
  2. Encodes it through Mimi (Speaker A = you)
  3. Silence-pads Speaker B (model will fill in with its "response")
  4. Builds a 1-row parquet matching prepare_dataset.py output format
  5. Saves to /workspace/my_voice/test.parquet

After this runs, use generate.py + decode_tokens.py on the parquet.

Usage:
  python test_my_voice.py --wav my_voice.wav [--prompt_seconds 10]
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders


def encode_with_mimi(wav_path: str, mimi: loaders.MimiModel, device: torch.device) -> np.ndarray:
    """Load a WAV, resample to Mimi rate, encode → audio tokens (8, T)."""
    wav, sr = torchaudio.load(wav_path)  # (1, samples) or (2, samples)
    if wav.shape[0] == 2:
        wav = wav.mean(dim=0, keepdim=True)  # mix stereo to mono
    if sr != mimi.sample_rate:
        wav = torchaudio.transforms.Resample(sr, mimi.sample_rate)(wav)
    wav = wav.to(device)
    with torch.no_grad():
        # Mimi expects (B, 1, samples)
        audio_ids = mimi.encode(wav.reshape(1, 1, -1))[0]  # (8, T)
    return audio_ids.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Path to your input WAV (any rate, will resample)")
    ap.add_argument("--output_dir", default="/workspace/my_voice", help="Where to save the parquet")
    ap.add_argument("--prompt_seconds", type=int, default=10, help="How long the prompt is (will be padded with silence on B)")
    ap.add_argument("--text_padding_id", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Mimi on {device}...")
    mimi = loaders.get_mimi(
        filename=hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors"),
        device=device,
    )

    print(f"Encoding {args.wav}...")
    audio_a = encode_with_mimi(args.wav, mimi, device)  # (8, T)
    K, T = audio_a.shape
    print(f"  → audio shape: ({K}, {T}), duration ≈ {T/12.5:.1f} sec")

    if T < args.prompt_seconds * 12.5:
        print(f"  WARNING: WAV is shorter than {args.prompt_seconds}s — using full length as prompt")

    # Cap at the prompt length so generate.py has room to generate
    prompt_frames = int(args.prompt_seconds * 12.5)
    if T > prompt_frames:
        audio_a = audio_a[:, :prompt_frames]
        T = prompt_frames
        print(f"  Truncated to {T} frames ({T/12.5:.1f} sec) for prompt")

    # Channel A: your voice
    text_a = np.full(T, args.text_padding_id, dtype=np.int32)
    a_stream = np.concatenate([text_a[None], audio_a], axis=0).astype(np.int32)  # (9, T)

    # Channel B: silence (will be filled in by model)
    # Mimi's silence tokens are typically 0 or a "neutral" code; for simplicity, use 0
    # The first audio codebook position usually has token 0 as silence
    audio_b = np.zeros((K, T), dtype=np.int32)
    text_b = np.full(T, args.text_padding_id, dtype=np.int32)
    b_stream = np.concatenate([text_b[None], audio_b], axis=0).astype(np.int32)  # (9, T)

    # Build dataframe matching prepare_dataset.py output
    df = pd.DataFrame([{
        "dialogue_id": "my_voice_test",
        "A": a_stream.tolist(),
        "B": b_stream.tolist(),
    }])

    out_path = os.path.join(args.output_dir, "my_voice-001-of-001.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved parquet to {out_path}")
    print(f"\nNow run generate.py with --eval_data_files {out_path}")


if __name__ == "__main__":
    main()
