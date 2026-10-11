from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from app.audio import read_reference, write_wav
from app.cli import main, read_text
from app.errors import OpenVoiceError
from app.tts import QwenTTS, choose_device


def test_reference_warns_on_silence_without_rejecting(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(16000), 16000)
    audio, rate, report = read_reference(path)
    assert audio.shape == (16000,)
    assert rate == 16000
    assert report.silence_fraction == 1
    assert any("quiet" in warning for warning in report.warnings)


def test_reference_stereo_and_original_preserved(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.full((32000, 2), 0.2), 16000)
    before = path.read_bytes()
    audio, _, report = read_reference(path)
    assert report.channels == 2 and report.duration_seconds == 2
    assert audio.ndim == 1
    assert before == path.read_bytes()


@pytest.mark.parametrize("contents", [b"", b"not an audio file"])
def test_corrupt_file(tmp_path, contents):
    path = tmp_path / "bad.wav"
    path.write_bytes(contents)
    with pytest.raises(OpenVoiceError, match="decoded"):
        read_reference(path)


def test_missing_reference(tmp_path):
    with pytest.raises(OpenVoiceError, match="missing"):
        read_reference(tmp_path / "missing.wav")


def test_reference_duration_limit(tmp_path):
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(8000 * 121), 8000)
    with pytest.raises(OpenVoiceError, match="120 seconds"):
        read_reference(path)


def test_export_valid_audio_and_never_overwrites(tmp_path):
    path = tmp_path / "result.wav"
    write_wav(path, np.array([-2, 0, 2]), 24000)
    audio, rate = sf.read(path)
    assert rate == 24000 and np.max(np.abs(audio)) <= 1
    before = path.read_bytes()
    with pytest.raises(OpenVoiceError, match="already exists"):
        write_wav(path, np.ones(20), 24000)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.partial"))


def test_failed_export_does_not_publish_partial_audio(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("codec failure")
    monkeypatch.setattr("app.audio.sf.write", fail)
    with pytest.raises(OpenVoiceError, match="save WAV"):
        write_wav(tmp_path / "result.wav", np.ones(100), 24000)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("audio", [np.array([]), np.array([np.nan]), np.zeros(100), np.ones((3, 2))])
def test_invalid_model_output_not_saved(tmp_path, audio):
    path = tmp_path / "result.wav"
    with pytest.raises(OpenVoiceError):
        write_wav(path, audio, 24000)
    assert not path.exists()


@pytest.mark.parametrize("requested,available,expected", [
    ("auto", False, "cpu"), ("auto", True, "cuda:0"),
    ("cuda", False, "cpu"), ("cpu", True, "cpu"),
])
def test_device_fallback(requested, available, expected):
    assert choose_device(requested, available) == expected


def test_missing_model_no_import_or_download(tmp_path):
    with pytest.raises(OpenVoiceError, match="missing"):
        QwenTTS(tmp_path).load(lambda _: None)


def test_adapter_uses_real_api_contract(tmp_path):
    backend = QwenTTS(tmp_path)
    model = Mock()
    model.create_voice_clone_prompt.return_value = ["prompt"]
    model.generate_voice_clone.return_value = ([np.ones(240)], 24000)
    backend.model = model
    audio = np.ones(16000, dtype=np.float32)
    result, rate = backend.generate(audio, 16000, "Reference.", "New sentence.", "English", lambda _: None)
    model.create_voice_clone_prompt.assert_called_once_with(ref_audio=(audio, 16000), ref_text="Reference.", x_vector_only_mode=False)
    model.generate_voice_clone.assert_called_once_with(text="New sentence.", language="English", voice_clone_prompt=["prompt"], max_new_tokens=2048, non_streaming_mode=False)
    assert rate == 24000 and len(result) == 240


def test_load_failure_recovers_to_unloaded(tmp_path, monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr('app.tts.windows_weight_reader', nullcontext)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").touch()
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), float32="float32", OutOfMemoryError=MemoryError)
    factory = Mock(side_effect=RuntimeError("broken model"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=factory)))
    backend = QwenTTS(tmp_path)
    with pytest.raises(OpenVoiceError, match="Could not load"):
        backend.load(lambda _: None)
    assert backend.model is None
    assert factory.call_args.kwargs["local_files_only"] is True
    assert factory.call_args.kwargs["dtype"] == "float32"


def test_transcript_nonempty_and_bounded(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("  ")
    with pytest.raises(OpenVoiceError):
        read_text(path, "Transcript", 10)
    path.write_text("x" * 11)
    with pytest.raises(OpenVoiceError):
        read_text(path, "Transcript", 10)


def test_permission_required_before_any_model_work(capsys):
    assert main(["clone", "--reference", "missing.wav", "--transcript-file", "ref.txt", "--text-file", "text.txt", "--output", "out.wav"]) == 1
    assert "permission" in capsys.readouterr().err


def test_cancel_returns_dedicated_status(monkeypatch):
    def interrupt(_):
        raise KeyboardInterrupt
    monkeypatch.setattr("app.cli.run", interrupt)
    assert main(["doctor"]) == 130
