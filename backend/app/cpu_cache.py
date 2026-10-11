"""Create a derived, sharded FP32 cache from verified official weights.

No quantization or model substitution: BF16 -> FP32 is an exact widening.
Storing that conversion on disk lets SafeTensors map FP32 weights at load time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError("Destination already exists; choose a new cache directory.")
    manifest = json.loads((source / "openvoice4u-manifest.json").read_text())
    if manifest["model_id"] != "Qwen/Qwen3-TTS-12Hz-1.7B-Base":
        raise ValueError("Expected the verified official Qwen Base model.")
    if shutil.disk_usage(source).free < 12 * 1024**3:
        raise ValueError("Keep 12 GiB free for the derived FP32 cache.")
    weight_files = list(source.rglob("*.safetensors"))
    if not weight_files or len({path.parent for path in weight_files}) != len(weight_files):
        raise ValueError("This converter expects one verified weight file per model component directory.")
    staging = destination.with_name(destination.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if ".cache" in relative.parts or not path.is_file() or path.suffix == ".safetensors":
            continue
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    hashes = {}
    for weights in weight_files:
        folder = staging / weights.parent.relative_to(source)
        folder.mkdir(parents=True, exist_ok=True)
        mapping = {}
        tensors = {}
        shard_bytes = total_bytes = count = 0

        def flush() -> None:
            nonlocal tensors, shard_bytes, count
            if not tensors:
                return
            count += 1
            filename = f"model-cpu-{count:05d}.safetensors"
            target = folder / filename
            save_file(tensors, target, metadata={"format": "pt"})
            for key in tensors:
                mapping[key] = filename
            with target.open("rb") as handle:
                hashes[str(target.relative_to(staging)).replace("\\", "/")] = hashlib.file_digest(handle, "sha256").hexdigest()
            tensors = {}
            shard_bytes = 0

        with safe_open(weights, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(torch.float32)
                size = tensor.numel() * tensor.element_size()
                if shard_bytes + size > 128 * 1024**2:
                    flush()
                tensors[key] = tensor.contiguous()
                shard_bytes += size
                total_bytes += size
                del tensor
            flush()
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": mapping}, indent=2), encoding="utf-8")
        print(f"Converted {weights.parent.name}: {len(mapping)} tensors, {count} shards, {total_bytes / 2**30:.2f} GiB", flush=True)
    manifest["derived_precision"] = "float32"
    manifest["derived_files_sha256"] = hashes
    (staging / "openvoice4u-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    staging.rename(destination)
    print(f"CPU cache ready: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
