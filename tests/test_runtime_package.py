import importlib.util
from pathlib import Path

import pytest


def builder():
    path = Path(__file__).resolve().parents[1] / 'scripts/bundle_runtime.py'
    spec = importlib.util.spec_from_file_location('runtime_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packaging_refuses_stale_runtime_without_modifying_it(tmp_path):
    output = tmp_path / 'runtime'
    output.mkdir()
    private = output / 'private-recording.wav'
    private.write_bytes(b'private fixture')
    with pytest.raises(RuntimeError, match='empty directory'):
        builder().ensure_empty_destination(output)
    assert private.read_bytes() == b'private fixture'


def test_packaging_accepts_new_or_empty_directory(tmp_path):
    output = tmp_path / 'runtime'
    builder().ensure_empty_destination(output)
    builder().ensure_empty_destination(output)
    assert list(output.iterdir()) == []


def test_packaging_rejects_file_destination(tmp_path):
    output = tmp_path / 'runtime'
    output.write_text('keep')
    with pytest.raises(RuntimeError, match='empty directory'):
        builder().ensure_empty_destination(output)
    assert output.read_text() == 'keep'
