"""prep_imagenet_modal.py: download ImageNet-1k from Hugging Face and write it to a Modal Volume as an ImageFolder.

Result: /vol/imagenet/train/<label:04d>/*.JPEG and /vol/imagenet/val/<label:04d>/*.JPEG. The folder names are the zero-padded
class ids, so ImageFolder's sorted order gives class index == label id (the order torchvision's pretrained weights use).

One-time setup:
  1. Accept the terms of the dataset at https://huggingface.co/datasets/ILSVRC/imagenet-1k (it is gated).
  2. modal secret create huggingface-secret HF_TOKEN=<your HF token>
  3. modal run --detach prep_imagenet_modal.py
It can be re-run: finished shards are skipped.

Files: ImageNet is ~1.33M files, over the 500,000-inode limit of Modal Volumes v1, so this uses a v2 volume.
"""
import os, modal

app = modal.App("imagenet-prep")
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub", "pyarrow")
vol = modal.Volume.from_name("imagenet-vol", create_if_missing=True, version=2)
REPO = "ILSVRC/imagenet-1k"
SPLITS = {"train": "train", "validation": "val"}              # HF split name -> folder name


def convert_parquet(parquet_path, out_root, split_dir, tag):
    """Write every image of one parquet shard to <out_root>/<split_dir>/<label:04d>/<tag>_<row>.JPEG. Returns the image count."""
    import pyarrow.parquet as pq
    n = 0
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=256, columns=["image", "label"]):
        for img, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            d = os.path.join(out_root, split_dir, f"{label:04d}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{tag}_{n}.JPEG"), "wb") as f:    # the parquet already holds the original JPEG bytes
                f.write(img["bytes"])
            n += 1
    return n


def process_shard(repo_path):
    from huggingface_hub import hf_hub_download
    out_root = "/vol/imagenet"
    tag = os.path.basename(repo_path).replace(".parquet", "")
    marker = os.path.join(out_root, ".done", tag)
    if os.path.exists(marker):
        return tag, -1                                         # already done
    split_dir = SPLITS[tag.split("-")[0]]
    local = hf_hub_download(REPO, repo_path, repo_type="dataset", local_dir="/tmp/hf")   # uses HF_TOKEN from the environment
    n = convert_parquet(local, out_root, split_dir, tag)
    os.remove(local)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, "w").write(str(n))
    return tag, n


@app.function(image=image, cpu=16, memory=32768, timeout=8 * 3600, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("huggingface-secret")])
def prep():
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
             if f.startswith("data/") and f.endswith(".parquet") and f.split("/")[-1].split("-")[0] in SPLITS]
    if not files:
        raise SystemExit(f"no data/train-*.parquet or data/validation-*.parquet found in {REPO}: the repo layout may have changed")
    print(f"{len(files)} shards", flush=True)
    counts = {"train": 0, "validation": 0}
    with ThreadPoolExecutor(max_workers=12) as ex:             # network and disk bound, so threads are enough
        for tag, n in ex.map(process_shard, sorted(files)):
            print(tag, "skipped" if n < 0 else n, flush=True)
            if n > 0: counts[tag.split("-")[0]] += n
    vol.commit()
    print("images written this run:", counts, "| expected 1,281,167 train and 50,000 validation in total")


@app.local_entrypoint()
def main():
    prep.remote()
