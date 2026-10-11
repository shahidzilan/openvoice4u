"""One bounded request per private desktop child process; stdout is NDJSON only."""
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

from .audio import read_reference, write_wav
from .config import LANGUAGES, MODEL_FOLDER
from .errors import OpenVoiceError
from .storage import Store, atomic_json

PROTOCOL = sys.stdout


def emit(kind: str, **fields):
    PROTOCOL.write(json.dumps(dict(kind=kind, **fields), ensure_ascii=False) + '\n')
    PROTOCOL.flush()


def progress(message: str):
    emit('progress', message=message)


def settings(store):
    path = store.root / 'settings.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'device': 'auto', 'models': str(store.root / 'models')}


def model_paths(store):
    base = Path(settings(store)['models']).resolve()
    return base / MODEL_FOLDER, base / (MODEL_FOLDER + '-cpu'), base / 'whisper'


def dispatch(operation, args, store):
    original, cpu, whisper_dir = model_paths(store)
    if operation == 'status':
        import psutil
        return dict(voices=store.voices(), settings=settings(store), languages=LANGUAGES,
                    ram_gib=round(psutil.virtual_memory().total / 2**30, 1),
                    available_gib=round(psutil.virtual_memory().available / 2**30, 1),
                    tts_ready=(cpu / 'openvoice4u-manifest.json').is_file(),
                    whisper_ready=(whisper_dir / 'base.pt').is_file(), root=str(store.root))
    if operation == 'settings':
        if args['device'] not in ('auto', 'cpu', 'cuda'):
            raise OpenVoiceError('Invalid compute device.')
        models = Path(args['models']).resolve()
        models.mkdir(parents=True, exist_ok=True)
        from .generation import lexicon
        preferences = dict(settings(store), device=args['device'], models=str(models))
        preferences['lexicon'] = lexicon(args.get('lexicon', preferences.get('lexicon', {})))
        preferences['completion_sound'] = args.get('completion_sound', preferences.get('completion_sound', True)) is True
        atomic_json(store.root / 'settings.json', preferences)
        return {}
    if operation == 'save_voice':
        return store.save(args['name'], args['transcript'], Path(args['source']), args.get('consent') is True, args.get('id'))
    if operation == 'edit_voice':
        return store.edit(args['id'], args['name'], args['transcript'])
    if operation == 'delete_voice':
        store.delete(args['id'])
        return {}
    if operation == 'import_recording':
        raw = base64.b64decode(args['audio'], validate=True)
        if len(raw) > 25 * 1024**2:
            raise OpenVoiceError('Recording is too large.')
        path = store.root / ('recording-' + uuid.uuid4().hex + '.wav')
        path.write_bytes(raw)
        try:
            read_reference(path)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return {'path': str(path)}
    if operation == 'validate':
        import soundfile as sf
        audio, rate, report = read_reference(Path(args['source']))
        preview = io.BytesIO()
        sf.write(preview, audio, rate, format='WAV', subtype='PCM_16')
        return dict(**report.to_dict(), audio=base64.b64encode(preview.getvalue()).decode('ascii'))
    if operation == 'transcribe':
        if not (whisper_dir / 'base.pt').is_file():
            raise OpenVoiceError('Install Whisper Base in Settings first, or enter the transcript manually.')
        import whisper
        expected = whisper._MODELS['base'].split('/')[-2]
        with (whisper_dir / 'base.pt').open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                raise OpenVoiceError('Whisper checksum failed. Repair the model in Settings.')
        audio, rate, _ = read_reference(Path(args['source']))
        import librosa
        audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
        progress('Loading Whisper Base…')
        model = whisper.load_model('base', device='cpu', download_root=str(whisper_dir))
        progress('Transcribing reference locally…')
        result = model.transcribe(audio, fp16=False, verbose=None)
        if not result['text'].strip():
            raise OpenVoiceError('No speech detected. Review the reference or enter its transcript.')
        return {'transcript': result['text'].strip()}
    if operation == 'setup_models':
        from .download import download_model
        from .cpu_cache import convert
        if args.get('model') == 'whisper':
            import whisper
            progress('Downloading and verifying Whisper Base…')
            whisper._download(whisper._MODELS['base'], str(whisper_dir), False)
        else:
            download_model(original, progress)
            if not cpu.exists():
                progress('Preparing the CPU cache. This needs about 8 GiB of additional disk space…')
                staging = cpu.with_name(cpu.name + '.building')
                if staging.exists():
                    import shutil
                    if staging.resolve().parent != cpu.resolve().parent or staging.is_symlink():
                        raise OpenVoiceError('Unsafe cache location.')
                    shutil.rmtree(staging)
                convert(original, cpu)
        return {}
    if operation == 'history':
        from .generation import history
        return history(store)
    if operation == 'generate':
        from .generation import generate
        return generate(args, store, settings(store), original, cpu, progress)
    if operation in ('preview', 'export'):
        if args.get('voice'):
            voice = store.get(args['voice'])
            source = store.folder(voice['id']) / voice['reference']
        else:
            source = Path(args['source']).resolve()
            if source.parent != store.root / 'generations' or source.suffix != '.wav':
                raise OpenVoiceError('Select an audio result created by this application.')
        if operation == 'preview' and source.stat().st_size > 40 * 1024**2:
            return {'audio': '', 'message': 'Recording saved. This large file can be exported below and played in your audio player.'}
        if operation == 'preview':
            return {'audio': base64.b64encode(source.read_bytes()).decode('ascii')}
        import soundfile as sf
        destination = Path(args['destination']).resolve()
        if destination.suffix.lower() not in ('.mp3', '.wav'):
            raise OpenVoiceError('Choose a WAV or MP3 filename.')
        descriptor, temp = tempfile.mkstemp(prefix='.openvoice4u-', suffix='.partial', dir=destination.parent)
        try:
            with os.fdopen(descriptor, 'wb') as handle, sf.SoundFile(source) as reader:
                options = dict(format='WAV', subtype='PCM_24') if destination.suffix.lower() == '.wav' else dict(
                    format='MP3', subtype='MPEG_LAYER_III', bitrate_mode='VARIABLE', compression_level=0.1)
                with sf.SoundFile(handle, mode='w', samplerate=reader.samplerate, channels=reader.channels, **options) as writer:
                    for block in reader.blocks(blocksize=65536, dtype='float32'):
                        writer.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temp, destination)
        except FileExistsError as error:
            raise OpenVoiceError('This file already exists. Choose a new filename.') from error
        finally:
            Path(temp).unlink(missing_ok=True)
        return {'path': str(destination)}
    raise OpenVoiceError('Unsupported operation.')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    try:
        line = sys.stdin.buffer.readline(40 * 1024**2)
        if len(line) >= 40 * 1024**2:
            raise OpenVoiceError('Request is too large.')
        request = json.loads(line)
        if request['operation'] == 'generate':
            from .process_job import contain_children
            contain_children()
        root = Path(os.environ.get('OPENVOICE4U_DATA_DIR', str(Path(os.environ['LOCALAPPDATA']) / 'OpenVoice4U')))
        with contextlib.redirect_stdout(sys.stderr):
            result = dispatch(request['operation'], request.get('args', {}), Store(root))
        emit('result', result=result)
    except OpenVoiceError as error:
        emit('error', message=str(error))
        return 1
    except Exception as error:
        emit('error', message=f'Operation failed ({type(error).__name__}). Check the input, model files, and available disk space; then retry.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
