import sys
from types import SimpleNamespace

import pytest

from app.low_memory import ReleasingDecoder


@pytest.mark.parametrize("fail", [False, True])
def test_weights_released_before_decoder_runs(monkeypatch, fail):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    progress = []
    core = SimpleNamespace(talker=object(), speaker_encoder=object())

    class Codec:
        sample_rate = 24000

        def decode(self, inputs):
            assert core.talker is None
            assert core.speaker_encoder is None
            if fail:
                raise RuntimeError("decoder failed")
            return inputs, self.sample_rate

    core.speech_tokenizer = Codec()
    decoder = ReleasingDecoder(core, progress.append, 0)
    if fail:
        with pytest.raises(RuntimeError, match="decoder failed"):
            decoder.decode([])
    else:
        assert decoder.decode([]) == ([], 24000)
    assert decoder.sample_rate == 24000
    assert "Decoding" in progress[-1]


def test_native_decoder_uses_smaller_context_aware_windows(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    calls = []
    decoder = SimpleNamespace(chunked_decode=lambda codes, **kwargs: calls.append(kwargs))
    codec = SimpleNamespace(model=SimpleNamespace(decoder=decoder), decode=lambda inputs: decoder.chunked_decode(inputs))
    core = SimpleNamespace(speech_tokenizer=codec, talker=object(), speaker_encoder=object())
    ReleasingDecoder(core, lambda _: None, 0).decode([])
    assert calls == [dict(chunk_size=75, left_context_size=25)]
