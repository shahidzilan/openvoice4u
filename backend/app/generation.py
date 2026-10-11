"""Bounded audio jobs: fresh inference process per segment and streaming output."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone

import numpy as np
import soundfile as sf

from .chunking import chunks
from .errors import OpenVoiceError
from .storage import atomic_json


def lexicon(value):
    if not isinstance(value, dict) or len(value) > 200:
        raise OpenVoiceError('Use at most 200 pronunciation replacements.')
    result = {}
    for key, replacement in value.items():
        if not isinstance(replacement, str) or not key.strip() or not replacement.strip() or max(len(key), len(replacement)) > 100:
            raise OpenVoiceError('Each pronunciation entry needs a word or phrase and a replacement (up to 100 characters).')
        if key.strip().casefold() in {k.casefold() for k in result}:
            raise OpenVoiceError('Pronunciation entries must be unique, ignoring case.')
        result[key.strip()] = replacement.strip()
    return result


def pronounce(text, replacements):
    mapping = {k.casefold(): v for k, v in lexicon(replacements).items()}
    if not mapping:
        return text
    pattern = r'(?<!\w)(?:' + '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r')(?!\w)'
    return re.sub(pattern, lambda match: mapping[match.group().casefold()], text, flags=re.IGNORECASE)


def history(store):
    path = store.root / 'generation_history.json'
    records = []
    if path.exists():
        try:
            records = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(records, list) or any(not isinstance(r, dict) or not all(k in r for k in ('path', 'timestamp', 'text_preview')) for r in records):
                raise ValueError('Invalid history')
        except (ValueError, TypeError):
            # Preserve damaged metadata and recover the audio inventory.
            path.rename(path.with_name('generation_history.corrupt-' + uuid.uuid4().hex + '.json'))
            records = []
    known = {Path(r['path']).name for r in records}
    changed = not path.exists()
    for audio in (store.root / 'generations').glob('*.wav'):
        if audio.name not in known and not audio.is_symlink():
            try:
                info = sf.info(audio)
            except (OSError, RuntimeError):
                continue
            records.append(dict(id=audio.stem, path=str(audio), timestamp=datetime.fromtimestamp(audio.stat().st_mtime, timezone.utc).isoformat(),
                                text_preview='Earlier generation (text not recorded)', voice_name='Saved audio', seconds=info.duration, chunks=0, deep_voice=False))
            changed = True
    if changed:
        atomic_json(path, records)
    result = []
    for record in records:
        audio = (store.root / 'generations' / Path(record['path']).name).resolve()
        if audio.parent == store.root / 'generations' and audio.suffix == '.wav':
            result.append(dict(record, path=str(audio), available=audio.is_file()))
    return sorted(result, key=lambda r: r['timestamp'], reverse=True)


def run_segment(payload, output, progress):
    # Explicit import path also works with the isolated portable Python runtime.
    bootstrap = "import sys;sys.path.insert(0,sys.argv[1]);from app.segment import main;main()"
    command = [sys.executable, '-I', '-u', '-X', 'faulthandler', '-c', bootstrap, str(Path(__file__).resolve().parents[1])]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    # Keep the latest diagnostic stack, never synthesis text or reference content.
    with (output.parent.parent / 'voice-worker.log').open('w', encoding='utf-8') as diagnostic, subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostic,
                          text=True, encoding='utf-8', creationflags=creationflags) as child:
        child.stdin.write(json.dumps(dict(payload, output=str(output)), ensure_ascii=False) + '\n')
        child.stdin.close()
        failure = None
        for line in child.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('kind') == 'progress':
                progress(event['message'])
            if event.get('kind') == 'error':
                failure = event['message']
        code = child.wait()
    if code or not output.is_file():
        raise OpenVoiceError(failure or f'Voice segment stopped (exit 0x{code & 0xffffffff:08X}). Close memory-heavy applications and retry. Previous completed recordings remain in History.')


def deepen(wave, rate):
    import librosa
    return librosa.effects.pitch_shift(wave, sr=rate, n_steps=-4)


def notify_complete():
    if os.name == 'nt':
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
        except (RuntimeError, OSError):
            pass  # A muted/unavailable sound device must not invalidate audio.


def generate(args, store, preferences, original, cpu, progress):
    from .config import LANGUAGES
    voice = store.get(args['voice'])
    language = args.get('language', 'Auto')
    if language not in LANGUAGES:
        raise OpenVoiceError('Unsupported language.')
    chunks(args['text'])  # Validate original input before replacement expansion.
    replacements = preferences.get('lexicon', {}) if args.get('pronunciation', True) else {}
    pieces = chunks(pronounce(args['text'], replacements))
    deep = args.get('deep_voice') is True
    identifier = uuid.uuid4().hex
    output = store.root / 'generations' / (identifier + '.wav')
    records = history(store)
    # Disk-backed segments and a streaming writer keep memory independent of text length.
    with tempfile.TemporaryDirectory(prefix='generation-', dir=store.root) as temporary:
        folder = Path(temporary)
        merged = folder / 'merged.wav'
        writer = None
        frames = 0
        try:
            for index, text in enumerate(pieces):
                prefix = f'Segment {index + 1} / {len(pieces)} · '
                part = folder / 'segment.wav'
                run_segment(dict(text=text, language=language, device=preferences['device'], original=str(original), cpu=str(cpu),
                                 reference=str(store.folder(voice['id']) / voice['reference']), transcript=voice['transcript']),
                            part, lambda message: progress(prefix + message))
                wave, rate = sf.read(part, dtype='float32')
                if deep:
                    progress(prefix + 'Applying Deep Voice pitch shift…')
                    wave = deepen(wave, rate)
                if not np.isfinite(wave).all() or not len(wave):
                    raise OpenVoiceError('The voice model returned invalid audio.')
                # Five-millisecond edge fades remove join clicks without cutting words.
                fade = min(int(rate * .005), len(wave) // 2)
                if fade:
                    wave[:fade] *= np.linspace(0, 1, fade)
                    wave[-fade:] *= np.linspace(1, 0, fade)
                peak = float(np.max(np.abs(wave)))
                if peak > .99:
                    wave *= .99 / peak
                if writer is None:
                    writer = sf.SoundFile(merged, mode='w', samplerate=rate, channels=1, subtype='PCM_24')
                if rate != writer.samplerate:
                    raise OpenVoiceError('Audio sample rate changed between segments.')
                if index:
                    silence = np.zeros(int(rate * .12), dtype=np.float32)
                    writer.write(silence)
                    frames += len(silence)
                writer.write(wave)
                frames += len(wave)
                del wave
                part.unlink()
        finally:
            if writer is not None:
                writer.close()
        progress('Saving completed recording and history…')
        os.replace(merged, output)
    record = dict(id=identifier, path=str(output), timestamp=datetime.now(timezone.utc).isoformat(),
                  text_preview=' '.join(args['text'].split())[:240], voice_name=voice['name'], voice_id=voice['id'],
                  seconds=frames / rate, chunks=len(pieces), deep_voice=deep)
    atomic_json(store.root / 'generation_history.json', [record] + records)
    if preferences.get('completion_sound', True):
        notify_complete()
    return record
