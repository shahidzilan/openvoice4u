"""Memory sequencing for the pinned qwen-tts 0.1.1 pipeline.

The verified wrapper finishes core.generate before calling speech_tokenizer.decode.
Its post-decode code only trims the reference prefix. This adapter releases the
finished talker/speaker modules at that boundary; the wrapper becomes single-use.
"""
import gc
from pathlib import Path
from typing import Any, Callable


class ReleasingDecoder:
    def __init__(self, core: Any, progress: Callable[[str], None], reference_frames: int,
                 checkpoint: Path | None = None):
        self.codec = core.speech_tokenizer
        self.core = core
        self.progress = progress
        self.reference_frames = reference_frames
        self.checkpoint = checkpoint

    def __getattr__(self, name: str) -> Any:
        return getattr(self.codec, name)

    def decode(self, inputs: list[dict], *args: Any, **kwargs: Any) -> Any:
        if self.checkpoint is not None:
            from safetensors.torch import save
            # Exclusive save: debug checkpoints are local biometric data, never logs.
            tensors = {f"audio_codes_{i}": item["audio_codes"].detach().cpu().contiguous()
                       for i, item in enumerate(inputs)}
            payload = save(tensors, metadata={"reference_frames": str(self.reference_frames)})
            with self.checkpoint.open("xb") as output:
                output.write(payload)
        self.progress("Speech codes generated. Releasing voice model before audio decoding...")
        self.core.talker = None
        self.core.speaker_encoder = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.progress("Decoding generated speech...")
        # Use the pinned tokenizer's own causal/context-aware chunked decoder.
        # Its default 300 frames creates large CPU convolution intermediates.
        decoder = getattr(getattr(self.codec, 'model', None), 'decoder', None)
        if decoder is not None and hasattr(decoder, 'chunked_decode'):
            from functools import partial
            decoder.chunked_decode = partial(decoder.chunked_decode, chunk_size=75, left_context_size=25)
        return self.codec.decode(inputs, *args, **kwargs)
