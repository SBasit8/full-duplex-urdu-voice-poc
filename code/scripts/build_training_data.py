# build_training_data.py
# Builds training_data/ with:
#   - tts/<session>/*.webm + mapping.tsv  (extracted from 36 zips)
#   - tts/manifest.tsv  (combined audio_path <tab> sentence)
#   - common_voice/clips  (symlink to original clips dir)
#   - common_voice/train_manifest.tsv  (train + speaker-filtered validated extras)

import os, csv, sys, zipfile, shutil
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path("/home/ubuntu/vs_code/moshi_urdu ")
TTS_SRC = ROOT / "1773070415599-_10 hrs Urdu Text to Speech Corpus"
CV_SRC  = ROOT / "1774212483838-cv-corpus-25.0-2026-03-09-ur" / "cv-corpus-25.0-2026-03-09" / "ur"

OUT = ROOT / "training_data"
OUT_TTS = OUT / "tts"
OUT_CV  = OUT / "common_voice"
OUT_TTS.mkdir(parents=True, exist_ok=True)
OUT_CV.mkdir(parents=True, exist_ok=True)

# -------- 1) Extract TTS zips --------
print("==> Extracting TTS zips...")
tts_manifest_rows = []
zips = sorted(p for p in TTS_SRC.iterdir() if p.suffix == ".zip")
for i, z in enumerate(zips, 1):
    session = z.stem.replace(" ", "_").replace("__", "_")
    dest = OUT_TTS / session
    if dest.exists() and any(dest.iterdir()):
        print(f"  [{i}/{len(zips)}] {session}  (already extracted, skipping)")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        print(f"  [{i}/{len(zips)}] {session}")
    mp = dest / "mapping.tsv"
    if mp.exists():
        with open(mp, encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                audio = dest / row["audio_filename"]
                tts_manifest_rows.append((str(audio.relative_to(OUT)), row["sentence"]))

with open(OUT_TTS / "manifest.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["audio_path", "sentence"])
    w.writerows(tts_manifest_rows)
print(f"TTS manifest: {len(tts_manifest_rows)} clips -> {OUT_TTS/'manifest.tsv'}")

# -------- 2) Common Voice: symlink clips --------
link = OUT_CV / "clips"
if not link.exists():
    os.symlink(CV_SRC / "clips", link)
    print(f"Symlinked clips -> {link}")

# -------- 3) Build filtered CV train manifest --------
print("==> Building CV train manifest...")

def load_tsv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))

train = load_tsv(CV_SRC / "train.tsv")
dev   = load_tsv(CV_SRC / "dev.tsv")
test  = load_tsv(CV_SRC / "test.tsv")
validated = load_tsv(CV_SRC / "validated.tsv")

eval_speakers = {r["client_id"] for r in dev} | {r["client_id"] for r in test}
eval_paths    = {r["path"]      for r in dev} | {r["path"]      for r in test}
train_paths   = {r["path"]      for r in train}

# durations
durs = {}
with open(CV_SRC / "clip_durations.tsv", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t"); next(r)
    for row in r:
        if len(row) >= 2 and row[1].isdigit():
            durs[row[0]] = int(row[1])

kept = []
# include official train as-is
for r in train:
    kept.append(r)
# add validated extras, excluding dev/test speakers and dev/test clip paths
n_extra = 0
for r in validated:
    if r["path"] in train_paths:     continue
    if r["path"] in eval_paths:      continue
    if r["client_id"] in eval_speakers: continue
    kept.append(r); n_extra += 1

fields = ["client_id","path","sentence","up_votes","down_votes","age","gender","locale","duration_ms"]
out_tsv = OUT_CV / "train_manifest.tsv"
with open(out_tsv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    for r in kept:
        w.writerow({k: r.get(k, "") for k in fields[:-1]} | {"duration_ms": durs.get(r["path"], "")})

total_ms = sum(durs.get(r["path"], 0) for r in kept)
print(f"CV train manifest: {len(kept)} clips "
      f"({len(train)} official train + {n_extra} validated extras), "
      f"{total_ms/1000/3600:.2f} hr -> {out_tsv}")

# -------- 4) Summary --------
tts_total = len(tts_manifest_rows)
print("\n=== DONE ===")
print(f"TTS clips:       {tts_total}  (~9.78 hr per filenames)")
print(f"CV train clips:  {len(kept)}  ({total_ms/1000/3600:.2f} hr)")
print(f"Output: {OUT}")
