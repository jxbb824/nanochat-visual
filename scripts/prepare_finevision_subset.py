"""
Prepare a local subset of the FineVisionMax dataset by downloading only
the first K parquet shards from the Hugging Face Hub.

This avoids downloading the full dataset and keeps everything in parquet form
under ~/.cache/nanochat/finevision_parquet.

FineVision will then load only from these local parquet files.
"""

import os

from huggingface_hub import HfApi, hf_hub_download

from nanochat.common import get_base_dir, print0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a local FineVision parquet subset")
    parser.add_argument(
        "--num_shards",
        type=int,
        default=8,
        help="Number of parquet shards to download from the hub (default: 8)",
    )
    args = parser.parse_args()

    base_dir = get_base_dir()
    parquet_dir = os.path.join(base_dir, "finevision_parquet")
    os.makedirs(parquet_dir, exist_ok=True)

    # If there are already parquet files here, reuse them
    existing = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    if existing:
        print0(f"Found existing parquet files in {parquet_dir}, reusing them.")
        return

    repo_id = "HuggingFaceM4/FineVisionMax"
    api = HfApi()
    print0(f"Listing parquet files in {repo_id} ...")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet") and "train" in f]
    if not parquet_files:
        parquet_files = [f for f in files if f.endswith(".parquet")]
    parquet_files = sorted(parquet_files)
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in dataset repo {repo_id}")

    selected = parquet_files[: args.num_shards]
    print0(f"Downloading {len(selected)} parquet shards into {parquet_dir} ...")
    for rel_path in selected:
        print0(f"  - {rel_path}")
        hf_hub_download(
            repo_id=repo_id,
            filename=rel_path,
            repo_type="dataset",
            local_dir=parquet_dir,
            local_dir_use_symlinks=False,
        )

    print0("Done. Local FineVision parquet subset is ready.")


if __name__ == "__main__":
    main()



