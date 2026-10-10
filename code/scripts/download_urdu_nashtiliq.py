# download_urdu_data.py

from datasets import load_dataset
import json

print("Downloading Urdu Dolly 15K (Nastaliq script)...")
ds = load_dataset("aaqibsaeed/databricks-dolly-15k-ur", split="train")

count = 0
with open("urdu_alpaca.jsonl", "w", encoding="utf-8") as f:
    for item in ds:
        instruction = item.get("instruction", "").strip()
        inp         = item.get("input", "").strip()
        output      = item.get("output", "").strip()

        if not instruction or not output:
            continue

        prompt = f"{instruction}\n{inp}" if inp else instruction

        f.write(json.dumps({
            "prompt": prompt,
            "response": output
        }, ensure_ascii=False) + "\n")
        count += 1

print(f"Done. {count} examples saved to urdu_alpaca.jsonl")