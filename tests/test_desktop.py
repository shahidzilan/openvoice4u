import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from app.chunking import chunks
from app.errors import OpenVoiceError
from app.storage import Store
from app.worker import dispatch


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / 'data')


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / 'reference.wav'
    sf.write(path, np.sin(np.arange(24000) / 20) * .2, 24000)
    return path


def test_profile_roundtrip_edit_replace_delete(store, sample):
    original = sample.read_bytes()
    voice = store.save('First', 'Reference words', sample, True)
    assert (store.folder(voice['id']) / voice['original']).read_bytes() == original
    store.edit(voice['id'], 'Renamed', 'Corrected words')
    assert store.get(voice['id'])['transcript'] == 'Corrected words'
    replacement = store.save('Renamed', 'Another reference', sample, True, voice['id'])
    assert replacement['reference'] != voice['reference']
    assert (store.folder(voice['id']) / voice['original']).exists()
    assert len(store.voices()) == 1
    store.delete(voice['id'])
    assert store.voices() == []
    assert sample.read_bytes() == original


def test_profile_requires_consent_and_rejects_traversal(store, sample):
    with pytest.raises(OpenVoiceError):
        store.save('My voice', 'Words', sample, False)
    for identifier in ('../outside', '', 'not-a-uuid'):
        with pytest.raises(OpenVoiceError):
            store.delete(identifier)


def test_chunking_keeps_words_and_paragraph_boundaries():
    text = 'A short sentence. ' * 50 + '\n\nAnother paragraph!'
    parts = chunks(text)
    assert all(len(part) <= 350 for part in parts)
    assert ' '.join(parts).split() == text.split()
    assert parts[-1] == 'Another paragraph!'
    with pytest.raises(OpenVoiceError):
        chunks('x' * 351)
    with pytest.raises(OpenVoiceError):
        chunks(' ')


def test_export_wav_mp3_and_no_overwrite(store, sample, tmp_path):
    generated = store.root / 'generations' / 'sample.wav'
    generated.write_bytes(sample.read_bytes())
    for extension in ('wav', 'mp3'):
        output = tmp_path / ('export.' + extension)
        dispatch('export', {'source': str(generated), 'destination': str(output)}, store)
        audio, rate = sf.read(output)
        assert rate == 24000 and len(audio) > 20000
        before = output.read_bytes()
        with pytest.raises(OpenVoiceError):
            dispatch('export', {'source': str(generated), 'destination': str(output)}, store)
        assert output.read_bytes() == before


def test_preview_rejects_arbitrary_files(store, sample):
    with pytest.raises(OpenVoiceError):
        dispatch('preview', {'source': str(sample)}, store)


def test_status_contains_no_transient_models(store):
    state = dispatch('status', {}, store)
    assert not state['tts_ready'] and not state['whisper_ready']
    assert state['voices'] == []

