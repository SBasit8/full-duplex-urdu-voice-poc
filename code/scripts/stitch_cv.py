# stitch_cv.py
# Group Common Voice clips by client_id, stitch with 300 ms silence into
# segments up to 90 s soft / 120 s hard cap, output 24 kHz mono 16-bit WAV.

import csv, json, os, sys, subprocess, tempfile
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

csv.field_size_limit(sys.maxsize)

ROOT       = Path("/home/ubuntu/vs_code/moshi_urdu ")
MANIFEST   = ROOT / "training_data/common_voice/train_manifest.tsv"
CLIPS_DIR  = ROOT / "training_data/common_voice/clips"   # symlink to real clips
OUT_DIR    = ROOT / "training_data/stitched_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR            = 24000
SOFT_CAP_MS   = 90_000
HARD_CAP_MS   = 120_000
SILENCE_MS    = 300
N_WORKERS     = 10

# --- 1) Generate a reusable 300 ms silence WAV (24 kHz mono 16-bit) ---
SILENCE_WAV = OUT_DIR / "_silence_300ms.wav"
if not SILENCE_WAV.exists():
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=mono:sample_rate={SR}",
        "-t", f"{SILENCE_MS/1000}", "-acodec", "pcm_s16le",
        str(SILENCE_WAV)
    ], check=True, capture_output=True)

# --- 2) Load manifest, group by speaker ---
by_speaker = defaultdict(list)
with open(MANIFEST, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        dur = int(row["duration_ms"]) if row["duration_ms"].isdigit() else 0
        if dur == 0: continue
        by_speaker[row["client_id"]].append({
            "path": row["path"], "sentence": row["sentence"], "dur_ms": dur,
        })

# --- 3) Pack each speaker's clips into segments (greedy, sort by path for stability) ---
def pack(clips):
    clips = sorted(clips, key=lambda c: c["path"])
    segs, cur, cur_ms = [], [], 0
    for c in clips:
        added_ms = c["dur_ms"] + (SILENCE_MS if cur else 0)
        if cur and cur_ms + added_ms > HARD_CAP_MS:
            segs.append(cur); cur, cur_ms = [], 0
            added_ms = c["dur_ms"]
        cur.append(c); cur_ms += added_ms
        if cur_ms >= SOFT_CAP_MS:
            segs.append(cur); cur, cur_ms = [], 0
    if cur: segs.append(cur)
    return segs

# --- 4) Worker: stitch one segment with ffmpeg concat demuxer ---
def stitch(job):
    cid, seg_idx, clips = job
    short = cid[:12]
    dst_dir = OUT_DIR / short
    dst_dir.mkdir(parents=True, exist_ok=True)
    wav_path = dst_dir / f"seg_{seg_idx:04d}.wav"
    json_path = dst_dir / f"seg_{seg_idx:04d}.json"

    if wav_path.exists() and json_path.exists():
        return (str(wav_path), "skip")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lf:
        for i, c in enumerate(clips):
            if i > 0:
                lf.write(f"file '{SILENCE_WAV}'\n")
            lf.write(f"file '{CLIPS_DIR / c['path']}'\n")
        list_file = lf.name

    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
            "-ar", str(SR), "-ac", "1", "-acodec", "pcm_s16le",
            str(wav_path)
        ], check=True, capture_output=True)
    finally:
        os.unlink(list_file)

    # Per-clip offsets within the stitched wav
    offsets, t = [], 0.0
    for i, c in enumerate(clips):
        if i > 0: t += SILENCE_MS / 1000
        offsets.append({
            "path": c["path"], "start_s": round(t, 3),
            "end_s": round(t + c["dur_ms"]/1000, 3),
            "sentence": c["sentence"],
        })
        t += c["dur_ms"] / 1000

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"client_id": cid, "duration_s": round(t, 3),
                   "num_clips": len(clips), "clips": offsets}, f, ensure_ascii=False)
    return (str(wav_path), f"{t:.1f}s/{len(clips)}")

# --- 5) Build job list ---
jobs = []
for cid, clips in by_speaker.items():
    for i, seg in enumerate(pack(clips)):
        jobs.append((cid, i, seg))
print(f"Speakers: {len(by_speaker)}  Segments to build: {len(jobs)}")

# --- 6) Run workers ---
done, skipped, errs = 0, 0, 0
with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = [ex.submit(stitch, j) for j in jobs]
    for i, fut in enumerate(as_completed(futs), 1):
        try:
            _, status = fut.result()
            if status == "skip": skipped += 1
            done += 1
        except Exception as e:
            errs += 1
            if errs < 5: print("ERR:", e)
        if i % 100 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)}  done={done} skip={skipped} err={errs}")

# --- 7) Build final manifest ---
print("Building manifest...")
rows = []
for cid_dir in sorted(OUT_DIR.iterdir()):
    if not cid_dir.is_dir(): continue
    for wav in sorted(cid_dir.glob("seg_*.wav")):
        jp = wav.with_suffix(".json")
        if not jp.exists(): continue
        meta = json.loads(jp.read_text(encoding="utf-8"))
        transcript = " ۔ ".join(c["sentence"] for c in meta["clips"])
        rows.append({
            "segment_path": str(wav.relative_to(OUT_DIR.parent)),
            "client_id":    meta["client_id"],
            "duration_s":   meta["duration_s"],
            "num_clips":    meta["num_clips"],
            "transcript":   transcript,
        })

man = OUT_DIR / "manifest.tsv"
with open(man, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["segment_path","client_id","duration_s","num_clips","transcript"], delimiter="\t")
    w.writeheader(); w.writerows(rows)

total_s = sum(r["duration_s"] for r in rows)
print(f"\n=== DONE ===")
print(f"Segments: {len(rows)}   Total: {total_s/3600:.2f} hr")
print(f"Manifest: {man}")
