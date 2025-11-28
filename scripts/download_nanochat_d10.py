"""
Download the nanochat d10 base run from Hugging Face and place its files
into the directories expected by this repo.

Source repo: https://huggingface.co/ThomasTheMaker/nanochat-d10
This repo mirrors the ~/.cache/nanochat layout created by speedrun.sh, but
here we only download and merge:
- base_checkpoints/**  (text base model)
- tokenizer/**         (matching tokenizer)
"""

import os

from huggingface_hub import snapshot_download

from nanochat.common import get_base_dir, print0


def move_tree(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)
    for root, dirs, files in os.walk(src_root):
        rel = os.path.relpath(root, src_root)
        dst_dir = os.path.join(dst_root, rel) if rel != "." else dst_root
        os.makedirs(dst_dir, exist_ok=True)
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(dst_dir, f)
            if os.path.exists(dst):
                print0(f"Found existing {dst}, skipping overwrite")
                continue
            print0(f"Copying {src} -> {dst}")
            os.replace(src, dst)


def main():
    base_dir = get_base_dir()
    tmp_dir = os.path.join(base_dir, "hf_nanochat_d10_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    print0(f"Downloading ThomasTheMaker/nanochat-d10 into {tmp_dir} ...")
    snapshot_download(
        repo_id="ThomasTheMaker/nanochat-d10",
        local_dir=tmp_dir,
        local_dir_use_symlinks=False,
        allow_patterns=["base_checkpoints/**", "tokenizer/**"],
    )

    # base_checkpoints tree (should contain d10 subdir) -> ~/.cache/nanochat/base_checkpoints
    src_base = os.path.join(tmp_dir, "base_checkpoints")
    if os.path.isdir(src_base):
        dst_base = os.path.join(base_dir, "base_checkpoints")
        move_tree(src_base, dst_base)

    # tokenizer -> ~/.cache/nanochat/tokenizer/d10
    src_tok = os.path.join(tmp_dir, "tokenizer")
    if os.path.isdir(src_tok):
        dst_tok = os.path.join(base_dir, "tokenizer", "d10")
        move_tree(src_tok, dst_tok)

    print0("Done. You can now load d10 with e.g. load_model(source='base', model_tag='d10').")
    print0("Remember to set NANOCHAT_TOKENIZER_DIR to the matching tokenizer, e.g.:")
    print0("  export NANOCHAT_TOKENIZER_DIR=$HOME/.cache/nanochat/tokenizer/d10")


if __name__ == "__main__":
    main()


