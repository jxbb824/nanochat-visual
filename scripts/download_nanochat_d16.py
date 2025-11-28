"""
Download the nanochat d16 base model from Hugging Face and place its files
into the directories expected by this repo.

Source repo: https://huggingface.co/0zk1/nanochat-d16-rocmrx9070-base

This repo is not a full ~/.cache dump but contains:
- meta_012800.json, model_012800.pt (nanochat-style checkpoint)
- tokenizer.pkl (+ possibly token_bytes.pt)

We will:
- put meta_*.json and model_*.pt under ~/.cache/nanochat/base_checkpoints/d16/
- put tokenizer.pkl (and token_bytes.pt if present) under ~/.cache/nanochat/tokenizer/d16/
"""

import os
import glob

from huggingface_hub import snapshot_download

from nanochat.common import get_base_dir, print0


def main():
    base_dir = get_base_dir()
    tmp_dir = os.path.join(base_dir, "hf_nanochat_d16_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    print0(f"Downloading 0zk1/nanochat-d16-rocmrx9070-base into {tmp_dir} ...")
    snapshot_download(
        repo_id="0zk1/nanochat-d16-rocmrx9070-base",
        local_dir=tmp_dir,
        local_dir_use_symlinks=False,
        allow_patterns=["model_*.pt", "meta_*.json", "tokenizer.pkl", "token_bytes.pt"],
    )

    # locate model_*.pt and meta_*.json
    model_files = glob.glob(os.path.join(tmp_dir, "model_*.pt"))
    meta_files = glob.glob(os.path.join(tmp_dir, "meta_*.json"))
    if not model_files or not meta_files:
        raise FileNotFoundError(f"Could not find model_*.pt and meta_*.json under {tmp_dir}")

    model_path = model_files[0]
    meta_path = meta_files[0]

    # move into base_checkpoints/d16
    d16_dir = os.path.join(base_dir, "base_checkpoints", "d16")
    os.makedirs(d16_dir, exist_ok=True)
    for src in [model_path, meta_path]:
        dst = os.path.join(d16_dir, os.path.basename(src))
        if os.path.exists(dst):
            print0(f"Found existing {dst}, skipping overwrite")
        else:
            print0(f"Moving {src} -> {dst}")
            os.replace(src, dst)

    # tokenizer artifacts -> tokenizer/d16
    tok_dir = os.path.join(base_dir, "tokenizer", "d16")
    os.makedirs(tok_dir, exist_ok=True)
    tok_src = os.path.join(tmp_dir, "tokenizer.pkl")
    if os.path.exists(tok_src):
        dst = os.path.join(tok_dir, "tokenizer.pkl")
        if os.path.exists(dst):
            print0(f"Found existing {dst}, skipping overwrite")
        else:
            print0(f"Moving {tok_src} -> {dst}")
            os.replace(tok_src, dst)
    tok_bytes_src = os.path.join(tmp_dir, "token_bytes.pt")
    if os.path.exists(tok_bytes_src):
        dst = os.path.join(tok_dir, "token_bytes.pt")
        if os.path.exists(dst):
            print0(f"Found existing {dst}, skipping overwrite")
        else:
            print0(f"Moving {tok_bytes_src} -> {dst}")
            os.replace(tok_bytes_src, dst)

    print0("Done. You can now load d16 with e.g. load_model(source='base', model_tag='d16').")
    print0("Remember to set NANOCHAT_TOKENIZER_DIR to the matching tokenizer, e.g.:")
    print0("  export NANOCHAT_TOKENIZER_DIR=$HOME/.cache/nanochat/tokenizer/d16")


if __name__ == "__main__":
    main()


