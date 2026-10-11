import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from . import __version__
from .config import LANGUAGES, MODEL_ID, model_directory
from .errors import OpenVoiceError


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_text(path: Path, label: str, limit: int) -> str:
    try:
        if path.stat().st_size > limit * 4 + 3:
            raise OpenVoiceError(f"{label} is too large for this prototype.")
        content = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as error:
        raise OpenVoiceError(f"Could not read {label}. Choose a UTF-8 text file.") from error
    if not content or len(content) > limit:
        raise OpenVoiceError(f"{label} must contain 1–{limit} characters for this prototype.")
    return content


def doctor(path: Path) -> dict:
    versions = {}
    for package in ("qwen-tts", "torch", "torchaudio", "transformers", "soundfile"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    result = {"application": "OpenVoice4U", "version": __version__,
              "python": sys.version.split()[0], "model_id": MODEL_ID,
              "model_directory": str(path),
              "model_files_present": (path / "config.json").is_file() and any((path / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")),
              "runtime": versions, "cuda_available": False}
    if versions["torch"] != "missing":
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["cpu_threads"] = torch.get_num_threads()
        try:
            import psutil
            memory = psutil.virtual_memory()
            result["ram_gib"] = round(memory.total / 2**30, 2)
            result["available_ram_gib"] = round(memory.available / 2**30, 2)
        except ImportError:
            pass
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="openvoice4u", description="OpenVoice4U — Local voice cloning, made simple. Research prototype.")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("doctor", "download"):
        command = commands.add_parser(name)
        command.add_argument("--model-dir", type=Path, default=model_directory())
    validate = commands.add_parser("validate", help="Inspect local reference audio; no model required")
    validate.add_argument("reference", type=Path)
    clone = commands.add_parser("clone", help="Generate a short Natural sample using the official Qwen Base model")
    clone.add_argument("--reference", type=Path, required=True)
    clone.add_argument("--transcript-file", type=Path, required=True)
    clone.add_argument("--text-file", type=Path, required=True)
    clone.add_argument("--output", type=Path, required=True)
    clone.add_argument("--model-dir", type=Path, default=model_directory())
    clone.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    clone.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    clone.add_argument("--cpu-safe-kernels", action="store_true", help="Diagnostic: disable oneDNN and use one CPU thread")
    clone.add_argument("--release-before-decode", action="store_true", help="Release the talker before decoding audio; single-use model")
    clone.add_argument("--codec-checkpoint", type=Path, help="Optional local debug checkpoint of generated speech codes")
    clone.add_argument("--language", choices=LANGUAGES, default="Auto")
    clone.add_argument("--confirm-permission", action="store_true",
                       help="Confirm you own the voice or have permission from the speaker to clone it")
    return root


def run(args: argparse.Namespace) -> None:
    if args.command == "doctor":
        print(json.dumps(doctor(args.model_dir), indent=2))
    elif args.command == "download":
        from .download import download_model
        download_model(args.model_dir, progress)
    elif args.command == "validate":
        from .audio import read_reference
        _, _, report = read_reference(args.reference)
        print(json.dumps(report.to_dict(), indent=2))
    elif args.command == "clone":
        if not args.confirm_permission:
            raise OpenVoiceError("Confirm you own this voice or have speaker permission with --confirm-permission.")
        if args.output.suffix.lower() != ".wav":
            raise OpenVoiceError("This prototype exports WAV. Choose a .wav output filename.")
        if args.output.exists():
            raise OpenVoiceError("Output already exists. Choose a new filename.")
        if not args.output.parent.is_dir():
            raise OpenVoiceError("Output folder does not exist. Create it before generating.")
        if args.codec_checkpoint is not None:
            if not args.release_before_decode:
                raise OpenVoiceError("A codec checkpoint requires --release-before-decode.")
            if args.codec_checkpoint.exists() or not args.codec_checkpoint.parent.is_dir():
                raise OpenVoiceError("Choose a new codec checkpoint filename in an existing folder.")
        transcript = read_text(args.transcript_file, "Reference transcript", 10000)
        text = read_text(args.text_file, "Target text", 300)
        from .audio import read_reference, write_wav
        from .tts import QwenTTS
        audio, rate, report = read_reference(args.reference)
        for warning in report.warnings:
            progress(warning)
        backend = QwenTTS(args.model_dir, args.device, args.precision, args.cpu_safe_kernels)
        started = time.monotonic()
        try:
            backend.load(progress)
            waveform, sample_rate = backend.generate(audio, rate, transcript, text, args.language, progress,
                release_before_decode=args.release_before_decode, codec_checkpoint=args.codec_checkpoint)
            write_wav(args.output, waveform, sample_rate)
            print(json.dumps({"output": str(args.output.resolve()), "model_id": MODEL_ID,
                              "device": backend.device, "sample_rate": sample_rate,
                              "audio_seconds": len(waveform) / sample_rate,
                              "elapsed_seconds": round(time.monotonic() - started, 2),
                              "identity_quality": "Requires human listening validation"}, indent=2))
        finally:
            backend.unload()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        progress("Cancelled. No completed result was reported.")
        return 130
    except OpenVoiceError as error:
        progress(f"OpenVoice4U: {error}")
        return 1
    except MemoryError:
        progress("OpenVoice4U ran out of memory. Close other applications and retry a shorter reference.")
        return 1
    except Exception as error:
        # Do not print exception strings: model libraries can include private input text.
        progress(f"OpenVoice4U: operation failed ({type(error).__name__}). Check the runtime, model files, and free memory.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
