from pathlib import Path
import os

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_FOLDER = "Qwen3-TTS-12Hz-1.7B-Base"
LANGUAGES = (
    "Auto", "Chinese", "English", "Japanese", "Korean", "German", "French",
    "Russian", "Portuguese", "Spanish", "Italian",
)


def model_directory() -> Path:
    override = os.environ.get("OPENVOICE4U_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
    return base / "OpenVoice4U" / "models" / MODEL_FOLDER
