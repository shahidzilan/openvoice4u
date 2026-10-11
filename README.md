# OpenVoice4U

**Turn your own voice sample into speech, locally on Windows.**

OpenVoice4U is an MIT-licensed desktop app built with React, Tauri and Python. Upload or record a voice sample, check its transcript, type new words, and export generated speech. Qwen3-TTS performs voice cloning; Whisper can transcribe the reference locally.

**Preview software:** quality depends on the reference and text. Long-form generation is supported through sequential segments, but large jobs can still fail. This is not a guarantee of studio quality or uninterrupted 4,000-word generation. See [validation](docs/validation.md).

**Publication status:** the review found unresolved dependency advisories. Source is available for experimental development; a stable installer release is not yet approved. See the [publication review](docs/release-review.md) for affected packages and remaining checks.

## Install on Windows

1. Open this repository's **Releases** page.
2. Download the Windows x64 `OpenVoice4U_<version>_x64-setup.exe` asset. GitHub's automatic **Source code** archives are for developers, not installers.
3. Run the installer, then launch OpenVoice4U from the Start menu.
4. In **Settings**, download the voice model and, optionally, Whisper Base. This first setup needs internet and substantial disk space. The installer does not contain model weights.

If Releases has no installer yet, the repository owner must run the included Windows build workflow and publish its output. Source code alone does not install the desktop app. Builds are unsigned unless the maintainer configures signing; Windows may display a publisher warning. Verify the download source and checksum before running it.

Tested platform: Windows 11 x64. The packaged app includes Python and its CPU runtime; end users do not need Python, Node or Rust. WebView2 is required; the Tauri installer handles its bootstrap if missing, which may need internet.

## Make your first voice

1. Download the voice model in **Settings**. Download Whisper Base too if you want automatic reference transcription.
2. Choose **Create voice**, then upload WAV, MP3, FLAC, M4A or OGG, or record with your microphone.
3. Use roughly 20–35 seconds of clear speech from one speaker, with no music or effects. An original WAV recording avoids extra lossy compression.
4. Listen and correct the transcript so it matches the actual spoken words. Without Whisper, enter it manually.
5. Name the voice and confirm that you own it or have the speaker's permission.
6. Select the saved voice, enter your script, choose its language, and click **Generate voice**.
7. Listen, then **Save WAV** or **Save MP3**. Completed clips also appear in **History**.

A sample is used as a reference for inference; the app does not train a new model for each user.

## Features

- Local voice cloning, optional local reference transcription, microphone recording and reusable voice profiles.
- Sentence-aware text splitting into segments up to 350 characters; each inference child exits before the next segment. Audio is joined on disk.
- Generation history with timestamps, text previews and file paths.
- Optional Deep Voice: a four-semitone pitch shift after synthesis. This changes timbre and is not a native model style instruction.
- Pronunciation replacements in Settings, with one `word = say as` entry per line.
- Optional Windows completion sound and cancellation of the worker process tree.
- WAV and MP3 export without overwriting existing files.

The text limit is 100,000 characters, not a promise that every maximum-size job completes. Files over 40 MiB can be exported and played externally; inline preview is bounded to avoid large memory copies.

## Hardware and storage

- 16 GB RAM was tested, but available memory and Windows virtual memory matter. More headroom is useful when other applications are open. A system-managed paging file may be necessary.
- Allow about 25 GB free during model setup: roughly 4.3 GiB of original weights, 7.8 GiB of derived CPU weights, the application runtime, and temporary working space. Whisper Base is about 139 MiB.
- CPU generation is slow: a few seconds of speech can take several minutes. Thousands of words may take many hours.
- The installer uses CPU PyTorch. Auto/CUDA falls back to CPU when CUDA is unavailable. CUDA packaging and GPU performance have not been validated for this release.

## Privacy

There are no accounts, cloud inference or analytics. Model setup downloads from official model sources. Inference is local after setup.

Installed user data is stored under `%LOCALAPPDATA%\OpenVoice4U`; Settings displays its actual location. Recordings, transcripts, generated speech, local diagnostic logs and `generation_history.json` are private user data. Reference originals are retained when replaced. Exported files stay where you save them.

A developer-created portable distribution can use a `portable-mode` marker and a sibling `data` folder. Never upload a used portable application folder to GitHub.

## Build from source

Install Python 3.12 x64, Node 22, Rust stable, Microsoft C++ Build Tools with the Windows SDK, and WebView2. From a fresh checkout:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-windows-cpu.lock.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pip check
npm ci
.venv\Scripts\python.exe -m pytest -q
npm run build
.venv\Scripts\python.exe scripts/bundle_runtime.py
npm run desktop
```

Build the Windows installer with `npm run tauri -- build`. It appears under `src-tauri/target/release/bundle/nsis`. Model weights and personal data are not bundled. Build scripts must run inside the project virtual environment when indicated. The Python lock file pins the tested environment but does not include download hashes.

The runtime packager requires an empty `release-runtime` directory. For subsequent builds, rename the previous directory before running it again. This prevents stale dependencies or private files from entering a new release.

## Publish and contribute

See [the GitHub publishing guide](docs/publishing.md), [contributing](CONTRIBUTING.md), [security](SECURITY.md), and [troubleshooting](docs/troubleshooting.md). The GitHub workflow builds a Windows installer on a clean runner; version tags create a draft prerelease for maintainer review.

The app's code is [MIT licensed](LICENSE). Dependencies and model weights retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Only clone permitted voices, and do not use the app for fraud or unauthorized impersonation.
