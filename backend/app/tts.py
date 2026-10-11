"""Adapter for the verified official Qwen Base API; no alternate engines."""
from pathlib import Path
from typing import Any, Callable
import gc
import os
from contextlib import contextmanager


@contextmanager
def windows_weight_reader():
    """Avoid Windows copy-on-write mmap commit overhead in pinned Transformers."""
    if os.name != 'nt':
        yield
        return
    import transformers.modeling_utils as modeling
    import safetensors.torch as tensor_io
    from functools import partial
    previous_model, previous_tensor = modeling.safe_open, tensor_io.safe_open
    modeling.safe_open = partial(previous_model, backend='pread')
    tensor_io.safe_open = partial(previous_tensor, backend='pread')
    try:
        yield
    finally:
        modeling.safe_open, tensor_io.safe_open = previous_model, previous_tensor

from .errors import OpenVoiceError


def choose_device(requested: str, cuda_available: bool) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise OpenVoiceError("Compute device must be auto, cpu, or cuda.")
    return "cuda:0" if requested != "cpu" and cuda_available else "cpu"


class QwenTTS:
    def __init__(self, model_path: Path, device: str = "auto", precision: str = "auto", cpu_safe_kernels: bool = False):
        self.model_path = model_path
        self.requested_device = device
        if precision not in {"auto", "float32", "bfloat16"}:
            raise OpenVoiceError("Precision must be auto, float32, or bfloat16.")
        self.requested_precision = precision
        self.cpu_safe_kernels = cpu_safe_kernels
        self.model: Any = None
        self.device: str | None = None

    def load(self, progress: Callable[[str], None]) -> None:
        if self.model is not None:
            return
        weights_present = any((self.model_path / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json"))
        if not (self.model_path / "config.json").is_file() or not weights_present:
            raise OpenVoiceError("Voice model is missing. Run the download command first.")
        # Set before importing Transformers: inference must never initiate downloads.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as error:
            raise OpenVoiceError("Voice runtime is missing. Install the project's [ml] dependencies.") from error
        self.device = choose_device(self.requested_device, torch.cuda.is_available())
        if self.device == "cpu" and self.cpu_safe_kernels:
            torch.backends.mkldnn.enabled = False
            torch.set_num_threads(1)
            progress("Using diagnostic CPU kernels (one thread, oneDNN disabled).")
        if self.requested_device == "cuda" and self.device == "cpu":
            progress("CUDA unavailable; using CPU.")
        precision = torch.float32 if self.device == "cpu" else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        if self.requested_precision != "auto":
            precision = getattr(torch, self.requested_precision)
        progress(f"Loading voice model on {self.device}...")
        try:
            with windows_weight_reader():
                self.model = Qwen3TTSModel.from_pretrained(
                    str(self.model_path.resolve()), device_map=self.device,
                    dtype=precision, attn_implementation="sdpa", local_files_only=True,
                )
        except (MemoryError, torch.OutOfMemoryError) as error:
            self.unload()
            raise OpenVoiceError("Model loading ran out of memory. Close other applications or try CPU.") from error
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.unload()
            if getattr(error, 'winerror', None) == 1455 or 'paging file is too small' in str(error).lower():
                raise OpenVoiceError('Windows has insufficient virtual memory to load the voice model. Close memory-heavy applications, or enable a system-managed paging file in Windows Advanced system settings, then retry. Your saved recordings are safe.') from error
            raise OpenVoiceError("Could not load the voice model. Check the runtime and repair the model download.") from error

    def generate(self, audio: Any, rate: int, transcript: str, text: str,
                 language: str, progress: Callable[[str], None], *,
                 release_before_decode: bool = False, codec_checkpoint: Path | None = None,
                 full_text_context: bool = False) -> tuple[Any, int]:
        if self.model is None:
            raise OpenVoiceError("Load the voice model before generating.")
        progress("Preparing voice reference...")
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=(audio, rate), ref_text=transcript, x_vector_only_mode=False,
        )
        if release_before_decode:
            from .low_memory import ReleasingDecoder
            core = self.model.model
            if not all(hasattr(core, name) for name in ("talker", "speaker_encoder", "speech_tokenizer")):
                raise OpenVoiceError("This Qwen runtime layout is not supported by Low Memory decoding.")
            reference_frames = int(prompt[0].ref_code.shape[0])
            core.speech_tokenizer = ReleasingDecoder(core, progress, reference_frames, codec_checkpoint)
        progress("Generating speech... CPU generation can take several minutes.")
        try:
            wavs, sample_rate = self.model.generate_voice_clone(
                text=text, language=language, voice_clone_prompt=prompt,
                non_streaming_mode=full_text_context,
                max_new_tokens=2048,
            )
        finally:
            if release_before_decode:
                self.unload()
        if len(wavs) != 1:
            raise OpenVoiceError("The voice model returned an unexpected number of outputs.")
        return wavs[0], sample_rate

    def unload(self) -> None:
        self.model = None
        gc.collect()
        # Import only if loading already imported torch.
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
