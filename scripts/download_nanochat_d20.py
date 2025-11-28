"""
Download the nanochat d20 run from Hugging Face and place its files
into the directories expected by this repo.

Source repo: https://huggingface.co/sampathchanda/nanochat-d20
This repo mirrors the ~/.cache/nanochat layout created by runs like speedrun.sh:
- base_checkpoints/d20
- mid_checkpoints/d20
- chatsft_checkpoints/d20
- chatrl_checkpoints/d20
- tokenizer/
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
    tmp_dir = os.path.join(base_dir, "hf_nanochat_d20_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    print0(f"Downloading sampathchanda/nanochat-d20 into {tmp_dir} ...")
    snapshot_download(
        repo_id="sampathchanda/nanochat-d20",
        local_dir=tmp_dir,
        local_dir_use_symlinks=False,
        allow_patterns=["chatsft_checkpoints/**", "tokenizer/**"],
    )

    # The repo already has base_checkpoints, mid_checkpoints, chatsft_checkpoints, chatrl_checkpoints, tokenizer
    # For FineVision we only need the SFT checkpoints and tokenizer.
    src_sft = os.path.join(tmp_dir, "chatsft_checkpoints")
    if os.path.isdir(src_sft):
        dst_sft = os.path.join(base_dir, "chatsft_checkpoints")
        move_tree(src_sft, dst_sft)

    # tokenizer is stored under tokenizer/d20 so multiple tokenizers can coexist
    src_tok = os.path.join(tmp_dir, "tokenizer")
    if os.path.isdir(src_tok):
        dst_tok = os.path.join(base_dir, "tokenizer", "d20")
        move_tree(src_tok, dst_tok)

    print0("Done. You can now load d20 with e.g. load_model(source='sft', model_tag='d20').")


if __name__ == "__main__":
    main()


