"""Versioned local profiles. Originals are immutable; metadata commits atomically."""
import json
import os
from pathlib import Path
import shutil
import uuid
from datetime import datetime, timezone

from .audio import read_reference, write_wav
from .errors import OpenVoiceError


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        for name in ('voices', 'generations', 'models'):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def folder(self, voice_id: str) -> Path:
        try:
            identifier = str(uuid.UUID(voice_id))
        except (ValueError, TypeError, AttributeError) as error:
            raise OpenVoiceError('Invalid voice identifier.') from error
        path = self.root / 'voices' / identifier
        if path.is_symlink() or not path.resolve().is_relative_to(self.root / 'voices'):
            raise OpenVoiceError('Invalid voice location.')
        return path

    def voices(self) -> list[dict]:
        result = []
        for path in (self.root / 'voices').glob('*/profile.json'):
            try:
                profile = self.get(path.parent.name)
                result.append(profile)
            except (OpenVoiceError, OSError, ValueError):
                continue
        return sorted(result, key=lambda item: item['name'].casefold())

    def get(self, voice_id: str) -> dict:
        folder = self.folder(voice_id)
        try:
            value = json.loads((folder / 'profile.json').read_text(encoding='utf-8'))
            if value['id'] != voice_id or value['schema'] != 1:
                raise ValueError()
            for key in ('original', 'reference'):
                if Path(value[key]).name != value[key]:
                    raise ValueError()
            return value
        except (OSError, ValueError, KeyError) as error:
            raise OpenVoiceError('This voice profile is missing or damaged.') from error

    def save(self, name: str, transcript: str, source: Path, consent: bool, voice_id: str | None = None) -> dict:
        if not consent:
            raise OpenVoiceError('Confirm that you own this voice or have permission to clone it.')
        if not name.strip() or len(name) > 80 or not transcript.strip() or len(transcript) > 10000:
            raise OpenVoiceError('Enter a voice name (up to 80 characters) and reference transcript (up to 10,000).')
        previous = self.get(voice_id) if voice_id else None
        identifier = voice_id or str(uuid.uuid4())
        folder = self.folder(identifier)
        folder.mkdir(exist_ok=True)
        audio, rate, report = read_reference(source)
        version = uuid.uuid4().hex
        original = folder / (version + source.suffix.lower())
        reference = folder / (version + '.wav')
        if original == reference:
            original = folder / (version + '-original.wav')
        now = datetime.now(timezone.utc).isoformat()
        try:
            shutil.copyfile(source, original)
            write_wav(reference, audio, rate)
            value = dict(schema=1, id=identifier, name=name.strip(), transcript=transcript.strip(),
                         original=original.name, reference=reference.name, report=report.to_dict(),
                         created=previous['created'] if previous else now, updated=now,
                         consent_at=now, model_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base')
            atomic_json(folder / 'profile.json', value)
        except Exception:
            original.unlink(missing_ok=True)
            reference.unlink(missing_ok=True)
            raise
        return value

    def edit(self, voice_id: str, name: str, transcript: str) -> dict:
        if not name.strip() or len(name) > 80 or not transcript.strip() or len(transcript) > 10000:
            raise OpenVoiceError('A valid name and reference transcript are required.')
        value = self.get(voice_id)
        value.update(name=name.strip(), transcript=transcript.strip(), updated=datetime.now(timezone.utc).isoformat())
        atomic_json(self.folder(voice_id) / 'profile.json', value)
        return value

    def delete(self, voice_id: str) -> None:
        self.get(voice_id)
        shutil.rmtree(self.folder(voice_id))
