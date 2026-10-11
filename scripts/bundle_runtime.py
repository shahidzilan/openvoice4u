"""Create a relocatable directory runtime from the tested Python 3.12 environment.

Uses CPython's isolated mode, excludes editable-install hooks and personal data,
and preserves package metadata and native-library licenses for redistribution.
"""
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

def ensure_empty_destination(destination):
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise RuntimeError('release-runtime must be an empty directory. Rename the previous build directory before rebuilding; never reuse an installed runtime.')
    destination.mkdir(exist_ok=True)


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / 'release-runtime'
    base = Path(sys.base_prefix)
    packages = Path(sys.prefix) / 'Lib' / 'site-packages'
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Run this script using the project .venv Python, not the global interpreter.')
    ensure_empty_destination(destination)
    for forbidden in ('data', 'voices', 'generations', 'models', 'portable-mode', 'settings.json', 'generation_history.json'):
        if (destination / forbidden).exists():
            raise RuntimeError(f'Refusing to package a runtime containing user data: {forbidden}')
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '__editable__*', 'openvoice4u-*.dist-info')
    for path in base.iterdir():
        if path.is_file() and path.suffix.lower() in ('.exe', '.dll', '.txt'):
            shutil.copy2(path, destination / path.name)
    for name in ('DLLs', 'Lib'):
        shutil.copytree(base / name, destination / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('site-packages', '__pycache__', '*.pyc'))
    shutil.copytree(packages, destination / 'Lib' / 'site-packages', dirs_exist_ok=True, ignore=ignore)
    shutil.copytree(root / 'backend' / 'app', destination / 'backend' / 'app', dirs_exist_ok=True, ignore=ignore)
    (destination / 'worker_entry.py').write_text(
        "from pathlib import Path\nimport sys, os\nruntime = Path(__file__).resolve().parent\n"
        "if (runtime.parent / 'portable-mode').is_file():\n    os.environ['OPENVOICE4U_DATA_DIR'] = str(runtime.parent / 'data')\n"
        "sys.path.insert(0, str(runtime / 'backend'))\n"
        "from app.worker import main\nraise SystemExit(main())\n", encoding='utf-8')
    inventory = sorted([{'name': d.metadata['Name'], 'version': d.version, 'license': d.metadata.get('License-Expression') or d.metadata.get('License', '')} for d in importlib.metadata.distributions() if d.metadata['Name'].lower() != 'openvoice4u'], key=lambda d:d['name'].lower())
    (destination / 'dependency-inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    shutil.copy2(root / 'LICENSE', destination / 'OpenVoice4U-LICENSE.txt')
    shutil.copy2(root / 'THIRD_PARTY_NOTICES.md', destination / 'THIRD_PARTY_NOTICES.md')
    print(f'Runtime prepared: {destination}')


if __name__ == '__main__':
    main()
