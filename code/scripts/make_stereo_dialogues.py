# make_stereo_dialogues.py
# Build synthetic 2-speaker stereo dialogues (Moshi fine-tune format):
#   L channel = Speaker A, R channel = Speaker B, turn-alternating, 24 kHz 16-bit.
# Pair speakers gender-aware (M<->F preferred, then random), one pair per speaker.

import csv, json, os, random, subprocess, sys, wave
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

csv.field_size_limit(sys.maxsize)
random.seed(42)

ROOT        = Path("/home/ubuntu/vs_code/moshi_urdu ")
MANIFEST    = ROOT / "training_data/common_voice/train_manifest.tsv"
CLIPS_DIR   = ROOT / "training_data/common_voice/clips"
OUT_DIR     = ROOT / "training_data/stereo_dialogues_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR              = 24000
GAP_MS          = 300
DLG_CAP_MS      = 120_000          # hard cap per dialogue wav
DLG_SOFT_MS     = 100_000          # start new dlg after this
N_WORKERS       = 8
GAP_SAMPLES     = int(SR * GAP_MS / 1000)

# ----- 1) Load manifest, group by speaker -----
spk_gender = {}
spk_clips  = defaultdict(list)
with open(MANIFEST, encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        d = int(r["duration_ms"]) if r["duration_ms"].isdigit() else 0
        if d == 0: continue
        cid = r["client_id"]
        spk_gender[cid] = r["gender"] or "unknown"
        spk_clips[cid].append({"path": r["path"], "dur_ms": d, "sentence": r["sentence"]})

speakers = list(spk_clips.keys())
males    = [s for s in speakers if spk_gender[s] == "male_masculine"]
females  = [s for s in speakers if spk_gender[s] == "female_feminine"]
others   = [s for s in speakers if spk_gender[s] == "unknown"]

random.shuffle(males); random.shuffle(females); random.shuffle(others)

# ----- 2) Build pairs: M<->F first, then random among leftovers -----
pairs = []
for i in range(min(len(males), len(females))):
    pairs.append((females[i], males[i], "MF"))
leftovers = males[len(females):] + others
random.shuffle(leftovers)
for i in range(0, len(leftovers) - 1, 2):
    pairs.append((leftovers[i], leftovers[i+1], "XX"))
# If odd leftover, drop it (can't pair alone)
print(f"Pairs: {len(pairs)}  (MF: {sum(1 for p in pairs if p[2]=='MF')}, XX: {sum(1 for p in pairs if p[2]=='XX')})")

# ----- 3) Decode MP3 to int16 PCM at 24 kHz mono -----
def decode_mp3_to_pcm(path):
    r = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path),
         "-ar", str(SR), "-ac", "1", "-f", "s16le", "-"],
        check=True, capture_output=True
    )
    return np.frombuffer(r.stdout, dtype=np.int16)

# ----- 4) Build dialogues for one pair -----
def build_pair(args):
    a_id, b_id, tag = args
    a_clips = sorted(spk_clips[a_id], key=lambda c: c["path"])
    b_clips = sorted(spk_clips[b_id], key=lambda c: c["path"])
    random.Random(hash((a_id, b_id)) & 0xffffffff).shuffle(a_clips)
    random.Random(hash((b_id, a_id)) & 0xffffffff).shuffle(b_clips)

    pair_name = f"{a_id[:8]}_{b_id[:8]}_{tag}"
    pair_dir = OUT_DIR / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    dialogues_meta = []
    ia = ib = 0
    dlg_idx = 0

    while ia < len(a_clips) and ib < len(b_clips):
        left_parts, right_parts, turns, dur_ms = [], [], [], 0
        turn = "A"
        t_ms = 0

        while dur_ms < DLG_SOFT_MS and ia < len(a_clips) and ib < len(b_clips):
            clip = a_clips[ia] if turn == "A" else b_clips[ib]
            # hard cap check before committing the turn
            projected = dur_ms + clip["dur_ms"] + (GAP_MS if turns else 0)
            if projected > DLG_CAP_MS: break

            # decode
            try:
                pcm = decode_mp3_to_pcm(CLIPS_DIR / clip["path"])
            except subprocess.CalledProcessError:
                # skip a broken clip
                if turn == "A": ia += 1
                else:           ib += 1
                continue

            gap = np.zeros(GAP_SAMPLES, dtype=np.int16) if turns else np.zeros(0, dtype=np.int16)
            silent_same_len = np.zeros(len(pcm), dtype=np.int16)

            if turn == "A":
                left_parts.extend([gap, pcm])
                right_parts.extend([gap, silent_same_len])
                ia += 1
            else:
                left_parts.extend([gap, silent_same_len])
                right_parts.extend([gap, pcm])
                ib += 1

            start_ms = t_ms + (GAP_MS if turns else 0)
            end_ms   = start_ms + int(len(pcm) * 1000 / SR)
            turns.append({"spk": turn, "path": clip["path"],
                          "start_s": round(start_ms/1000, 3),
                          "end_s":   round(end_ms/1000, 3),
                          "sentence": clip["sentence"]})
            t_ms   = end_ms
            dur_ms = t_ms
            turn   = "B" if turn == "A" else "A"

        if not turns: break

        L = np.concatenate(left_parts)
        R = np.concatenate(right_parts)
        assert len(L) == len(R)
        interleaved = np.empty(len(L) * 2, dtype=np.int16)
        interleaved[0::2] = L
        interleaved[1::2] = R

        wav_path = pair_dir / f"dlg_{dlg_idx:04d}.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(interleaved.tobytes())

        meta = {"pair": pair_name, "speaker_A": a_id, "speaker_B": b_id,
                "duration_s": round(len(L)/SR, 3), "num_turns": len(turns),
                "turns": turns}
        with open(wav_path.with_suffix(".json"), "w", encoding="utf-8") as jf:
            json.dump(meta, jf, ensure_ascii=False)

        dialogues_meta.append((str(wav_path.relative_to(OUT_DIR.parent)),
                               a_id, b_id, meta["duration_s"], len(turns),
                               " ".join(f"[{t['spk']}] {t['sentence']}" for t in turns)))
        dlg_idx += 1

    return dialogues_meta

# ----- 5) Run -----
print("Building dialogues...")
all_rows = []
with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = [ex.submit(build_pair, p) for p in pairs]
    for i, fut in enumerate(as_completed(futs), 1):
        rows = fut.result()
        all_rows.extend(rows)
        total = sum(r[3] for r in all_rows)
        print(f"  pair {i}/{len(pairs)}  dialogues so far: {len(all_rows)}  audio: {total/3600:.2f} hr")

# ----- 6) Manifest -----
man = OUT_DIR / "manifest.tsv"
with open(man, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["dialogue_path","speaker_A","speaker_B","duration_s","num_turns","transcript"])
    w.writerows(all_rows)

total_s = sum(r[3] for r in all_rows)
print(f"\n=== DONE ===")
print(f"Pairs: {len(pairs)}   Dialogues: {len(all_rows)}   Total: {total_s/3600:.2f} hr")
print(f"Manifest: {man}")
