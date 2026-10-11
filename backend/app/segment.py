"""Private inference child; its process exits before the next segment starts."""
import contextlib
import json
from pathlib import Path
import sys


def main():
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    from .worker import progress, emit
    from .audio import read_reference, write_wav
    from .errors import OpenVoiceError
    try:
        args = json.loads(sys.stdin.readline())
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from .tts import QwenTTS, choose_device
            actual = choose_device(args['device'], torch.cuda.is_available())
            path = Path(args['cpu'] if actual == 'cpu' else args['original'])
            if not (path / 'openvoice4u-manifest.json').is_file():
                raise OpenVoiceError('Prepare the voice model in Settings first.')
            audio, rate, _ = read_reference(Path(args['reference']))
            model = QwenTTS(path, device=args['device'], cpu_safe_kernels=True)
            try:
                model.load(progress)
                wave, output_rate = model.generate(audio, rate, args['transcript'], args['text'], args['language'], progress,
                                                   release_before_decode=True, full_text_context=True)
                write_wav(Path(args['output']), wave, output_rate)
            finally:
                model.unload()
    except Exception as error:
        emit('error', message=str(error) if isinstance(error, OpenVoiceError) else f'Voice segment failed ({type(error).__name__}).')
        raise SystemExit(1)
