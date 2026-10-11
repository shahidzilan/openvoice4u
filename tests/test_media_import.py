import numpy as np
import pytest
from app.audio import read_reference
from app.errors import OpenVoiceError


def test_m4a_local_decode(tmp_path):
    import av
    path = tmp_path / 'reference.m4a'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('aac', rate=24000)
        stream.layout = 'mono'
        wave = (np.sin(np.arange(24000, dtype=np.float32) / 20) * .2).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(wave, format='fltp', layout='mono')
        frame.sample_rate = 24000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    original = path.read_bytes()
    audio, rate, report = read_reference(path)
    assert rate == 48000 and .9 < report.duration_seconds < 1.2
    assert np.isfinite(audio).all() and report.rms > .1
    assert path.read_bytes() == original


def test_corrupt_m4a_is_readable_error(tmp_path):
    path = tmp_path / 'broken.m4a'
    path.write_bytes(b'not an audio file')
    with pytest.raises(OpenVoiceError, match='could not be decoded'):
        read_reference(path)
