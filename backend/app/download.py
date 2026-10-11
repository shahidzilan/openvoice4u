"""Explicit, resumable downloads from the official model repository only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Callable

from .config import MODEL_ID
from .errors import OpenVoiceError


def download_model(destination: Path, progress: Callable[[str], None]) -> None:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # This command is the only operation allowed to access the model hub.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise OpenVoiceError("Install the project's [ml] dependencies before downloading models.") from error
    progress("Checking official model metadata...")
    info = HfApi().model_info(MODEL_ID, files_metadata=True)
    files = [entry for entry in info.siblings if not entry.rfilename.startswith(".")]
    for entry in files:
        if not (destination / entry.rfilename).resolve().is_relative_to(destination.resolve()):
            raise OpenVoiceError("Model repository contains an invalid file path.")
    total = sum(entry.size or 0 for entry in files)
    destination.mkdir(parents=True, exist_ok=True)
    # Conservative: reserve full snapshot size plus 1 GiB, even on repair.
    if shutil.disk_usage(destination).free < total + 1024**3:
        raise OpenVoiceError(f"Not enough disk space. Keep at least {(total + 1024**3) / 1024**3:.1f} GiB free.")
    progress(f"Downloading {total / 1024**3:.2f} GiB from {MODEL_ID}, revision {info.sha}...")
    snapshot_download(repo_id=MODEL_ID, revision=info.sha, local_dir=destination,
                      allow_patterns=[entry.rfilename for entry in files], max_workers=2)
    progress("Verifying downloaded files...")
    verified = 0
    for entry in files:
        path = (destination / entry.rfilename).resolve()
        if not path.is_relative_to(destination.resolve()):
            raise OpenVoiceError("Model repository contains an invalid file path.")
        if not path.is_file() or (entry.size is not None and path.stat().st_size != entry.size):
            raise OpenVoiceError("Model download is incomplete. Retry the download command.")
        if entry.lfs and entry.lfs.sha256:
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != entry.lfs.sha256:
                raise OpenVoiceError(f"Checksum failed for {entry.rfilename}. Remove that file and retry.")
            verified += 1
    manifest = {"model_id": MODEL_ID, "revision": info.sha, "bytes": total,
                "sha256_verified_files": verified}
    temporary = destination / "openvoice4u-manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(destination / "openvoice4u-manifest.json")
    progress("Voice model is ready.")
