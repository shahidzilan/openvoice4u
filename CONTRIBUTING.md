# Contributing

Follow the Windows setup in README. Run the backend tests and frontend build before submitting a pull request. Do not download models in ordinary tests; real-model experiments must be opt-in and record hardware, model revision, duration and memory without committing private audio.

Keep model-specific code in the adapter. Use verified upstream APIs, preserve original recordings, retain permission checks and atomic writes, and avoid silent model substitutions. Keep changes focused on local, permitted voice cloning.

Before packaging, run scripts/check_release.py and prepare the runtime from a clean virtual environment. Never commit models, runtimes, generated audio, reference transcripts, local diagnostic files or screenshots containing user data. Use scripts/export_source.py to create a reviewed source snapshot. See docs/publishing.md for installer builds and release validation.
