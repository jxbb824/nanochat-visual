"""
Download the released nanochat d34 chat model from Hugging Face and place
its files into the directories expected by nanochat.

This follows the instructions from the author (slightly adapted):
- token_bytes.pt, tokenizer.pkl -> ~/.cache/nanochat/tokenizer/d34
- meta_169150.json, model_169150.pt -> ~/.cache/nanochat/chatsft_checkpoints/d34/

Model card: https://huggingface.co/karpathy/nanochat-d34
Discussion: https://github.com/karpathy/nanochat/discussions/314
"""

import os

from huggingface_hub import snapshot_download

from nanochat.common import get_base_dir, print0


def main():
    base_dir = get_base_dir()
    tmp_dir = os.path.join(base_dir, "hf_nanochat_d34_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    print0(f"Downloading karpathy/nanochat-d34 into {tmp_dir} ...")
    snapshot_download(
        repo_id="karpathy/nanochat-d34",
        local_dir=tmp_dir,
        local_dir_use_symlinks=False,
    )

    # Target locations:
    # tokenizer is stored under tokenizer/d34 so multiple tokenizers can coexist
    tokenizer_dir = os.path.join(base_dir, "tokenizer", "d34")
    os.makedirs(tokenizer_dir, exist_ok=True)
    chatsft_dir = os.path.join(base_dir, "chatsft_checkpoints", "d34")
    os.makedirs(chatsft_dir, exist_ok=True)

    def move_if_exists(filename, dst_dir):
        for root, _, files in os.walk(tmp_dir):
            if filename in files:
                src = os.path.join(root, filename)
                dst = os.path.join(dst_dir, filename)
                if os.path.exists(dst):
                    print0(f"Found existing {dst}, skipping overwrite")
                    return True
                print0(f"Moving {src} -> {dst}")
                os.replace(src, dst)
                return True
        return False

    move_if_exists("token_bytes.pt", tokenizer_dir)
    move_if_exists("tokenizer.pkl", tokenizer_dir)
    move_if_exists("meta_169150.json", chatsft_dir)
    move_if_exists("model_169150.pt", chatsft_dir)

    print0("Done. You can now load it with load_model(source='sft', model_tag='d34').")


if __name__ == "__main__":
    main()


