import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.download import download_model
from app.errors import OpenVoiceError


def setup_hub(monkeypatch, tmp_path, *, content=b"model", expected=b"model", free=10**10):
    entry = SimpleNamespace(rfilename="model.safetensors", size=len(expected),
                            lfs=SimpleNamespace(sha256=hashlib.sha256(expected).hexdigest()))
    info = SimpleNamespace(sha="revision123", siblings=[entry])
    api = Mock()
    api.model_info.return_value = info
    def snapshot(**kwargs):
        (tmp_path / "model.safetensors").write_bytes(content)
    download = Mock(side_effect=snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api, snapshot_download=download))
    monkeypatch.setattr("app.download.shutil.disk_usage", lambda _: SimpleNamespace(free=free))
    return download


def test_official_snapshot_pinned_and_verified(tmp_path, monkeypatch):
    download = setup_hub(monkeypatch, tmp_path)
    download_model(tmp_path, lambda _: None)
    assert download.call_args.kwargs["repo_id"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert download.call_args.kwargs["revision"] == "revision123"
    manifest = json.loads((tmp_path / "openvoice4u-manifest.json").read_text())
    assert manifest["sha256_verified_files"] == 1


def test_low_disk_prevents_download(tmp_path, monkeypatch):
    download = setup_hub(monkeypatch, tmp_path, free=1)
    with pytest.raises(OpenVoiceError, match="disk space"):
        download_model(tmp_path, lambda _: None)
    download.assert_not_called()


def test_checksum_failure_not_marked_installed(tmp_path, monkeypatch):
    setup_hub(monkeypatch, tmp_path, content=b"wrong")
    with pytest.raises(OpenVoiceError, match="Checksum"):
        download_model(tmp_path, lambda _: None)
    assert not (tmp_path / "openvoice4u-manifest.json").exists()


def test_interrupted_download_can_be_retried(tmp_path, monkeypatch):
    download = setup_hub(monkeypatch, tmp_path)
    original = download.side_effect
    download.side_effect = ConnectionError("interrupted")
    with pytest.raises(ConnectionError):
        download_model(tmp_path, lambda _: None)
    assert not (tmp_path / "openvoice4u-manifest.json").exists()
    download.side_effect = original
    download_model(tmp_path, lambda _: None)
    assert (tmp_path / "openvoice4u-manifest.json").exists()
