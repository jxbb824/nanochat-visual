"""
finevision_loader.py

从 HuggingFaceM4/FineVision 直接下载 parquet 分片，
并解析为 (image, text) 对，不使用 datasets.load_dataset。

同时提供：
- 按图片 id 构建 train/val split
- 按 split 过滤后，返回纯文本 batch（List[str]），
  用于复用 nano 原来的 token 流逻辑。

依赖:
    pip install requests pyarrow pillow tqdm
"""

import os
import io
import json
import random
import hashlib
from typing import Iterator, Tuple, List, Dict, Any, Optional, Set

import requests
from tqdm import tqdm
from PIL import Image
import pyarrow.parquet as pq

NUM_TRAIN_SHARDS = {
    "CoSyn_400k_chart": 52,
}

BASE_URL = (
    "https://huggingface.co/datasets/HuggingFaceM4/FineVision/resolve/main"
)


def _hf_headers() -> Dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

def download_finevision_parquets(
    subset: str,
    root: str = "./finevision_data",
    max_shards: Optional[int] = None,
) -> List[str]:
    if subset not in NUM_TRAIN_SHARDS:
        raise ValueError(
            f"Unknown subset {subset}. Please add it to NUM_TRAIN_SHARDS."
        )

    total_shards = NUM_TRAIN_SHARDS[subset]
    if max_shards is None:
        num_shards = total_shards
    else:
        num_shards = min(max_shards, total_shards)

    subset_dir = os.path.join(root, subset)
    os.makedirs(subset_dir, exist_ok=True)

    local_paths: List[str] = []
    headers = _hf_headers()

    for i in range(num_shards):
        fname = f"train-{i:05d}-of-{num_shards:05d}.parquet"
        url = f"{BASE_URL}/{subset}/{fname}"
        local_path = os.path.join(subset_dir, fname)
        local_paths.append(local_path)

        if os.path.exists(local_path):
            print(f"[Skip] {local_path} already exists.")
            continue

        print(f"[Download] {url}")
        resp = requests.get(url, stream=True, headers=headers)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0) or 0)
        with open(local_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {fname}",
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return local_paths


def _decode_image_entry(entry: Any) -> Image.Image:
    if isinstance(entry, list):
        if not entry:
            raise ValueError("Empty images list in entry")
        entry = entry[0]

    if isinstance(entry, dict):
        if "bytes" in entry:
            img_bytes = entry["bytes"]
        else:
            raise ValueError(f"Unknown image dict schema: keys = {entry.keys()}")
    elif isinstance(entry, (bytes, bytearray, memoryview)):
        img_bytes = bytes(entry)
    else:
        raise TypeError(f"Unsupported image entry type: {type(entry)}")

    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _get_image_id(entry: Any) -> str:
    if isinstance(entry, list):
        if not entry:
            raise ValueError("Empty images list in entry")
        entry = entry[0]

    if isinstance(entry, dict):
        img_bytes = entry.get("bytes", None)
        path = str(entry.get("path", ""))
        if img_bytes is None:
            raise ValueError(f"images entry has no 'bytes' field: keys={entry.keys()}")
        h = hashlib.sha1(img_bytes).hexdigest()
        return f"{path}:{h}"

    s = repr(entry)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _format_text_entry(text_turns: Any) -> str:
    if not isinstance(text_turns, list):
        return str(text_turns)

    parts: List[str] = []
    for t in text_turns:
        if isinstance(t, dict) and "user" in t:
            parts.append(t["user"])
        else:
            parts.append(str(t))
    return "\n".join(parts)

def iter_finevision_pairs(
    subset: str,
    root: str = "./finevision_data",
) -> Iterator[Tuple[Image.Image, str]]:
    parquet_paths = download_finevision_parquets(subset, root)

    for path in parquet_paths:
        print(f"[Read] {path}")
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            images_col = rg.column("images").to_pylist()
            texts_col = rg.column("texts").to_pylist()

            assert len(images_col) == len(texts_col)
            for img_entry, text_entry in zip(images_col, texts_col):
                img = _decode_image_entry(img_entry)
                text = _format_text_entry(text_entry)
                yield img, text


def load_finevision(
    subset: str,
    root: str = "./finevision_data",
    max_samples: Optional[int] = None,
) -> List[Tuple[Image.Image, str]]:
    data: List[Tuple[Image.Image, str]] = []
    for i, (img, text) in enumerate(iter_finevision_pairs(subset, root)):
        data.append((img, text))
        if max_samples is not None and i + 1 >= max_samples:
            break
    print(f"[Done] Loaded {len(data)} samples from {subset}")
    return data

def build_image_split(
    subset: str,
    root: str = "./finevision_data",
    train_ratio: float = 0.98,
    seed: int = 42,
    max_shards: Optional[int] = None,
) -> Dict[str, Set[str]]:
    subset_dir = os.path.join(root, subset)
    os.makedirs(subset_dir, exist_ok=True)
    split_path = os.path.join(subset_dir, f"image_split_seed{seed}.json")

    if os.path.exists(split_path):
        print(f"[Split] Load existing split from {split_path}")
        with open(split_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return {
            "train": set(obj["train"]),
            "val": set(obj["val"]),
        }

    parquet_paths = download_finevision_parquets(subset, root, max_shards=max_shards)
    image_ids: Set[str] = set()

    print("[Split] Scanning image ids...")
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            images_col = rg.column("images").to_pylist()
            for img_entry in images_col:
                img_id = _get_image_id(img_entry)
                image_ids.add(img_id)

    image_ids_list = list(image_ids)
    random.Random(seed).shuffle(image_ids_list)
    n_train = int(len(image_ids_list) * train_ratio)
    train_ids = image_ids_list[:n_train]
    val_ids = image_ids_list[n_train:]

    print(
        f"[Split] total images={len(image_ids_list)}, "
        f"train={len(train_ids)}, val={len(val_ids)}"
    )

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {"train": train_ids, "val": val_ids},
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "train": set(train_ids),
        "val": set(val_ids),
    }


def finevision_text_batches(
    subset: str,
    split: str,
    root: str = "./finevision_data",
    start: int = 0,
    step: int = 1,
    max_shards: Optional[int] = None,
) -> Iterator[Tuple[List[Image.Image], List[str]]]:

    assert split in {"train", "val"}

    parquet_paths = download_finevision_parquets(subset, root, max_shards=max_shards)
    split_info = build_image_split(subset, root, max_shards=max_shards)
    allowed_ids: Set[str] = split_info[split]

    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            images_col = rg.column("images").to_pylist()
            texts_col = rg.column("texts").to_pylist()

            assert len(images_col) == len(texts_col)

            batch_images: List[Image.Image] = []
            batch_texts: List[str] = []

            for img_entry, text_entry in zip(images_col, texts_col):
                img_id = _get_image_id(img_entry)
                if img_id not in allowed_ids:
                    continue

                img = _decode_image_entry(img_entry)
                txt = _format_text_entry(text_entry)

                batch_images.append(img)
                batch_texts.append(txt)

            if batch_images:
                yield batch_images, batch_texts


if __name__ == "__main__":
    ds = load_finevision("CoSyn_400k_chart", max_samples=3)
    for i, (img, text) in enumerate(ds):
        print("=" * 80)
        print(f"Sample {i}: image size = {img.size}, mode = {img.mode}")
        print("Text preview:")
        print(text[:400] + ("..." if len(text) > 400 else ""))

    print("\n--- Build split & show a train text batch ---")
    for image_batch, text_batch in finevision_text_batches(
        "CoSyn_400k_chart",
        split="train",
        start=0,
        step=1,
    ):
        print(f"Batch contains {len(image_batch)} images")

        print(f"Batch contains {len(text_batch)} texts")

        print("First text preview:")
        print(text_batch[0][:300] + ("..." if len(text_batch[0]) > 300 else ""))

        img = image_batch[0]
        print(f"First image info: size={img.size}, mode={img.mode}")

        img.save("debug_image.png")
        print("Saved first image as debug_image.png")

        break
