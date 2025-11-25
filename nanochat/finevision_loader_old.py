import os
import io
from typing import Iterator, Tuple, List, Dict, Any, Optional

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
) -> List[str]:
    if subset not in NUM_TRAIN_SHARDS:
        raise ValueError(
            f"Unknown subset {subset}. Please add it to NUM_TRAIN_SHARDS."
        )

    num_shards = NUM_TRAIN_SHARDS[subset]
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


if __name__ == "__main__":
    ds = load_finevision("CoSyn_400k_chart", max_samples=3)
    for i, (img, text) in enumerate(ds):
        print("=" * 80)
        print(f"Sample {i}: image size = {img.size}, mode = {img.mode}")
        print("Text preview:")
        print(text[:400] + ("..." if len(text) > 400 else ""))
