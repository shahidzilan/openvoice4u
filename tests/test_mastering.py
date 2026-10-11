import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from app.chunking import chunks
from app.errors import OpenVoiceError
from app.mastering import master_narration


def test_short_sentences_are_not_split_to_fill_a_previous_chunk():
    first = 'one ' * 20 + 'ends.'
    second = 'two ' * 70 + 'ends.'
    parts = chunks(first + ' ' + second)
    assert parts == [first, second]


def test_master_is_separate_bounded_and_preserves_duration(tmp_path):
    rate = 24000
    time = np.arange(rate*8)/rate
    wave = (.15*np.sin(time*2*np.pi*180)*(0.4+0.6*np.sin(time*2*np.pi*.5)**2)).astype('float32')
    source = tmp_path/'source.wav'
    target = tmp_path/'master.wav'
    sf.write(source, wave, rate, subtype='PCM_24')
    before = source.read_bytes()
    result = master_narration(source, target)
    output, output_rate = sf.read(target)
    assert source.read_bytes() == before
    assert output_rate == 48000
    assert abs(len(output)/output_rate-8) < .025
    assert np.isfinite(output).all()
    assert np.max(abs(resample_poly(output, 4, 1))) < 10**(-1.2/20)
    assert result['target_lufs'] == -16
    with pytest.raises(OpenVoiceError):
        master_narration(source, target)
