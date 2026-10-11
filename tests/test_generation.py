import json
import numpy as np
import pytest
import soundfile as sf

from app import generation
from app.chunking import chunks
from app.errors import OpenVoiceError
from app.storage import Store
from app.worker import dispatch


def test_four_thousand_words_are_bounded_and_preserved():
    text = ' '.join(f'word{i}' for i in range(4000))
    parts = chunks(text)
    assert len(parts) > 80
    assert all(0 < len(p) <= 350 for p in parts)
    assert ' '.join(parts) == text


def test_pronunciation_is_literal_longest_whole_word_single_pass():
    assert generation.pronounce('SQL sql SQLite New York York C++!',
                                {'SQL': 'sequel', 'New York': 'new york city', 'York': 'yorkshire', 'C++': 'see plus plus'}) == 'sequel sequel SQLite new york city yorkshire see plus plus!'
    assert generation.pronounce('A', {'A': 'B', 'B': 'C'}) == 'B'
    with pytest.raises(OpenVoiceError):
        generation.lexicon({'Word': 'one', 'word': 'two'})


def test_deep_voice_lowers_pitch_without_changing_duration():
    rate = 24000
    wave = (.25 * np.sin(2*np.pi*240*np.arange(rate)/rate)).astype('float32')
    shifted = generation.deepen(wave, rate)
    frequency = np.argmax(abs(np.fft.rfft(shifted[3000:-3000]))) * rate / len(shifted[3000:-3000])
    assert len(shifted) == len(wave)
    assert abs(frequency - 240 * 2**(-4/12)) < 3
    assert np.isfinite(shifted).all()


@pytest.fixture
def prepared(tmp_path):
    store = Store(tmp_path / 'data')
    sample = tmp_path / 'ref.wav'
    sf.write(sample, np.sin(np.arange(24000)/20)*.2, 24000)
    voice = store.save('Test voice', 'Reference', sample, True)
    return store, voice


def test_long_job_streams_segments_and_commits_history_and_notifies(prepared, monkeypatch):
    store, voice = prepared
    calls, alerts = [], []
    def fake(payload, output, progress):
        assert not output.exists()
        calls.append(payload['text'])
        sf.write(output, np.ones(2400, dtype=np.float32)*.1, 24000)
    monkeypatch.setattr(generation, 'run_segment', fake)
    monkeypatch.setattr(generation, 'notify_complete', lambda: alerts.append(True))
    text = 'Hello world. ' * 2000
    r = generation.generate(dict(voice=voice['id'], text=text), store, dict(device='cpu'), store.root, store.root, lambda _: None)
    assert ' '.join(calls).split() == text.split()
    assert r['chunks'] == len(calls)
    assert sf.info(r['path']).frames == len(calls)*2400+(len(calls)-1)*2880
    assert alerts == [True]
    assert generation.history(Store(store.root))[0]['text_preview'] == r['text_preview']
    assert json.loads((store.root/'generation_history.json').read_text())[0]['path'] == r['path']
    assert not list(store.root.glob('generation-*'))


def test_failed_job_does_not_publish_or_notify(prepared, monkeypatch):
    store, voice = prepared
    def fail(*args):
        raise OpenVoiceError('Native failure')
    monkeypatch.setattr(generation, 'run_segment', fail)
    monkeypatch.setattr(generation, 'notify_complete', lambda: pytest.fail('Must not notify on failure'))
    with pytest.raises(OpenVoiceError, match='Native failure'):
        generation.generate(dict(voice=voice['id'], text='Hello.'), store, dict(device='cpu'), store.root, store.root, lambda _: None)
    assert list((store.root/'generations').iterdir()) == []
    assert generation.history(store) == []
    assert not list(store.root.glob('generation-*'))


def test_history_recovers_earlier_files_and_corrupt_metadata(prepared):
    store, voice = prepared
    audio = store.root/'generations'/'earlier.wav'
    sf.write(audio, np.ones(2400)*.1, 24000)
    (store.root/'generation_history.json').write_text('broken')
    records = generation.history(store)
    assert len(records) == 1 and records[0]['available']
    assert list(store.root.glob('generation_history.corrupt-*.json'))
    audio.unlink()
    assert generation.history(store)[0]['available'] is False


def test_preferences_persist_dictionary_and_notification(prepared):
    store, _ = prepared
    dispatch('settings', dict(device='cpu', models=str(store.root/'models'), lexicon={'SQL':'sequel'}, completion_sound=False), store)
    dispatch('settings', dict(device='cpu', models=str(store.root/'models')), store)
    settings = dispatch('status', {}, store)['settings']
    assert settings['lexicon'] == {'SQL':'sequel'}
    assert settings['completion_sound'] is False


def test_large_recording_exports_without_full_waveform_read(prepared, tmp_path, monkeypatch):
    store, _ = prepared
    source = store.root / 'generations' / 'large.wav'
    block = np.full(65536, .1, dtype=np.float32)
    with sf.SoundFile(source, mode='w', samplerate=24000, channels=1, subtype='PCM_24') as writer:
        for _ in range(220):
            writer.write(block)
    assert source.stat().st_size > 40 * 1024**2
    preview = dispatch('preview', {'source': str(source)}, store)
    assert preview['audio'] == '' and 'exported' in preview['message']
    monkeypatch.setattr(sf, 'read', lambda *a, **k: pytest.fail('Must stream instead of reading the complete file'))
    destination = tmp_path / 'large-export.wav'
    dispatch('export', {'source': str(source), 'destination': str(destination)}, store)
    assert sf.info(destination).frames == 220 * len(block)
