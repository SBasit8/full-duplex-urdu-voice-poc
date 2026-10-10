"""Merge sharded safetensors into a single model.safetensors file.

The MoshiForFinetuning loader expects a single safetensors file, but
DeepSpeed Zero 3 saves in HuggingFace sharded format (multiple files +
index.json). This script merges them.
"""
import json
import os
import sys
from safetensors.torch import load_file, save_file


def main(base_dir: str):
    idx_path = os.path.join(base_dir, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        # Already merged or never sharded
        single = os.path.join(base_dir, "model.safetensors")
        if os.path.exists(single):
            print(f"Already merged: {single} ({os.path.getsize(single)/1e9:.2f} GB)")
            return
        print(f"ERROR: no index.json and no model.safetensors at {base_dir}")
        sys.exit(1)

    idx = json.load(open(idx_path))
    weight_map = idx["weight_map"]

    shards = {}
    for fname in set(weight_map.values()):
        path = os.path.join(base_dir, fname)
        print(f"Loading {fname} ({os.path.getsize(path)/1e9:.2f} GB)...")
        shards[fname] = load_file(path)

    state_dict = {k: shards[v][k] for k, v in weight_map.items()}
    print(f"Merged {len(state_dict)} tensors")

    out = os.path.join(base_dir, "model.safetensors")
    print(f"Saving to {out}...")
    save_file(state_dict, out)
    print(f"Done. Size: {os.path.getsize(out)/1e9:.2f} GB")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/workspace/output/moshi-urdu-finetuned/step_110"
    main(base)
