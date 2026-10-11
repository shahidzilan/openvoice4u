from dataclasses import asdict, dataclass
from pathlib import Path
import os
import tempfile

import numpy as np
import soundfile as sf

from .errors import OpenVoiceError

# Application bounds, not purported Qwen model limits.
MAX_REFERENCE_SECONDS = 120
MAX_INPUT_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class AudioReport:
    duration_seconds: float
    sample_rate: int
    channels: int
    rms: float
    peak: float
    silence_fraction: float
    clipping_fraction: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def read_reference(path: Path) -> tuple[np.ndarray, int, AudioReport]:
    if not path.is_file():
        raise OpenVoiceError("Reference audio is missing. Choose a local audio file.")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise OpenVoiceError("Reference is too large. Use a speech excerpt under 100 MiB.")
    try:
        if path.suffix.lower() in {'.m4a', '.mp4', '.webm'}:
            import av
            import io
            blocks = []
            frames = 0
            # A file object prevents the decoder from treating a path as a network URL.
            container_format = 'matroska' if path.suffix.lower() == '.webm' else 'mov'
            with path.open('rb') as handle, av.open(handle, format=container_format,
                    options={'protocol_whitelist': 'file,pipe', 'enable_drefs': '0'}) as container:
                resampler = av.AudioResampler(format='fltp', layout='mono', rate=48000)
                for decoded in container.decode(audio=0):
                    for frame in resampler.resample(decoded):
                        frames += frame.samples
                        if frames > 48000 * MAX_REFERENCE_SECONDS:
                            raise OpenVoiceError('Use a reference of at most 120 seconds.')
                        blocks.append(frame.to_ndarray().reshape(-1))
                for frame in resampler.resample(None):
                    frames += frame.samples
                    blocks.append(frame.to_ndarray().reshape(-1))
            if not blocks or frames > 48000 * MAX_REFERENCE_SECONDS:
                raise OpenVoiceError('Use a non-empty reference of at most 120 seconds.')
            decoded_buffer = io.BytesIO()
            sf.write(decoded_buffer, np.concatenate(blocks), 48000, format='WAV', subtype='FLOAT')
            decoded_buffer.seek(0)
            audio_source = decoded_buffer
        else:
            audio_source = path
        with sf.SoundFile(audio_source) as source:
            rate, channels = source.samplerate, source.channels
            duration = source.frames / rate
            if not source.frames or duration > MAX_REFERENCE_SECONDS or channels > 8:
                raise OpenVoiceError("Use a non-empty reference of at most 120 seconds and 8 channels.")
            if not 8000 <= rate <= 192000:
                raise OpenVoiceError("Use reference audio sampled between 8 kHz and 192 kHz.")
            samples = source.read(dtype="float32", always_2d=True)
    except (RuntimeError, ValueError, OSError) as error:
        raise OpenVoiceError("Audio could not be decoded. Try a valid WAV, MP3, FLAC, M4A, or OGG recording.") from error
    if not samples.size or not np.isfinite(samples).all():
        raise OpenVoiceError("Reference audio is empty or contains invalid samples.")
    peak = float(np.max(np.abs(samples)))
    clipping = float(np.mean(np.abs(samples) >= 0.999))
    mono = np.mean(samples, axis=1, dtype=np.float32)
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    frame = max(1, int(rate * 0.02))
    padded = np.pad(mono, (0, (-mono.size) % frame))
    levels = np.sqrt(np.mean(padded.reshape(-1, frame).astype(np.float64) ** 2, axis=1))
    silence = float(np.mean(levels < 0.0031623))  # -50 dBFS, 20 ms windows
    warnings: list[str] = []
    if duration < 3:
        warnings.append("Very short reference; a longer clean speech recording may improve identity.")
    if duration > 40:
        warnings.append("Long reference increases inference cost; about 30 seconds is recommended.")
    if silence > 0.5:
        warnings.append("Much of this recording is quiet. Check for silence or missing speech.")
    if rms < 0.01:
        warnings.append("Reference volume is very low.")
    if clipping > 0.01:
        warnings.append("Reference may be clipped. Re-record at a lower input level.")
    if peak > 1:
        mono = mono / peak
        warnings.append("A processed in-memory copy was attenuated to avoid out-of-range amplitude.")
    return mono, rate, AudioReport(duration, rate, channels, rms, peak, silence, clipping, tuple(warnings))


def write_wav(path: Path, waveform: np.ndarray, rate: int) -> None:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all() or rate <= 0:
        raise OpenVoiceError("The model returned invalid audio; no output was saved.")
    peak = float(np.max(np.abs(waveform)))
    if peak <= 1e-8:
        raise OpenVoiceError("The model returned silent audio; no output was saved.")
    if peak > 1:
        waveform = waveform * (0.99 / peak)
    temporary: Path | None = None
    try:
        descriptor, filename = tempfile.mkstemp(prefix=".openvoice4u-", suffix=".partial", dir=path.parent)
        temporary = Path(filename)
        with os.fdopen(descriptor, "wb") as output:
            sf.write(output, waveform, rate, format="WAV", subtype="PCM_24")
            output.flush()
            os.fsync(output.fileno())
        # Atomic, no-replace publication on Windows NTFS and other hardlink-capable volumes.
        os.link(temporary, path)
    except FileExistsError as error:
        raise OpenVoiceError("Output already exists. Choose a new filename to preserve it.") from error
    except (OSError, RuntimeError, ValueError) as error:
        raise OpenVoiceError("Could not save WAV. Check free disk space and use a writable NTFS output folder.") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
