# Third-party software and models

The MIT license in LICENSE covers OpenVoice4U's original application code. It does not replace the licenses of its dependencies or model weights.

- Qwen3-TTS code and the official Qwen3-TTS-12Hz-1.7B-Base model: Apache-2.0. Model files are downloaded separately from [the official repository](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).
- Whisper: MIT; [official source](https://github.com/openai/whisper).
- CPython: Python Software Foundation licensing; its license files are copied into the runtime.
- PyTorch, NumPy, SciPy, PyAV and other Python packages retain their own licenses and native-library notices. The runtime preserves installed package metadata and license directories and includes dependency-inventory.json.
- FFmpeg libraries bundled with PyAV and libsndfile bundled with SoundFile have their own licensing requirements. Library build configuration matters; PyAV's own license alone does not describe all codec-library obligations.
- Frontend and Rust dependency versions are recorded in package-lock.json and src-tauri/Cargo.lock. Their licenses continue to apply.

Before distributing a binary, review the actual built dependency inventory, native-library notices and any applicable corresponding-source obligations. The inventory is metadata, not a complete legal compliance determination. No signing certificate or model license is included with the application's MIT grant.
