"""Optional, disk-streamed narration mastering; never a substitute for good speech."""
import os
from pathlib import Path
import tempfile
from fractions import Fraction

import av
import numpy as np
import soundfile as sf

from .errors import OpenVoiceError


def master_narration(source: Path, destination: Path):
    """Produce a separate 48 kHz PCM24 master using the bundled FFmpeg filters.

    A conservative rumble filter, gentle compression and -16 LUFS / -1.5 dBTP
    target are starting points for narration mixing, not a YouTube specification.
    No denoising, pitch manipulation, silence removal or invented detail.
    """
    destination = Path(destination)
    if destination.exists():
        raise OpenVoiceError('The comparison file already exists. Choose another filename.')
    descriptor, temporary = tempfile.mkstemp(prefix='.master-', suffix='.wav', dir=destination.parent)
    os.close(descriptor)
    try:
        with sf.SoundFile(source) as reader:
            if reader.channels != 1:
                raise OpenVoiceError('Narration mastering currently expects mono audio.')
            graph = av.filter.Graph()
            origin = graph.add_abuffer(sample_rate=reader.samplerate, format='fltp', layout='mono', time_base=Fraction(1, reader.samplerate))
            filters = [origin, graph.add('highpass', 'f=60:p=2'),
                       graph.add('acompressor', 'threshold=0.125:ratio=2:attack=15:release=150:makeup=1:knee=2'),
                       graph.add('loudnorm', 'I=-16:TP=-1.5:LRA=11'),
                       graph.add('aformat', 'sample_fmts=fltp:sample_rates=48000:channel_layouts=mono'),
                       graph.add('abuffersink')]
            graph.link_nodes(*filters)
            graph.configure()
            frames = 0
            position = 0
            with sf.SoundFile(temporary, mode='w', samplerate=48000, channels=1, subtype='PCM_24') as writer:
                def drain():
                    nonlocal frames
                    while True:
                        try:
                            frame = graph.pull()
                        except (av.error.BlockingIOError, av.error.EOFError):
                            break
                        wave = frame.to_ndarray().reshape(-1)
                        if not np.isfinite(wave).all():
                            raise OpenVoiceError('Mastering returned invalid audio.')
                        writer.write(wave)
                        frames += len(wave)
                for block in reader.blocks(blocksize=16384, dtype='float32'):
                    frame = av.AudioFrame.from_ndarray(block.reshape(1, -1), format='fltp', layout='mono')
                    frame.sample_rate = reader.samplerate
                    frame.time_base = Fraction(1, reader.samplerate)
                    frame.pts = position
                    position += len(block)
                    origin.push(frame)
                    drain()
                origin.push(None)
                drain()
        if not frames:
            raise OpenVoiceError('No audio was produced by mastering.')
        os.link(temporary, destination)
        return {'path': str(destination), 'seconds': frames / 48000, 'sample_rate': 48000,
                'target_lufs': -16, 'target_true_peak_dbtp': -1.5}
    finally:
        Path(temporary).unlink(missing_ok=True)
