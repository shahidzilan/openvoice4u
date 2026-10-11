"""Fail builds with inconsistent versions or user data in the packaged runtime."""
import json
from pathlib import Path
import sys
import tomllib

root = Path(__file__).resolve().parents[1]
versions = [json.loads((root / p).read_text())['version'] for p in ('package.json', 'src-tauri/tauri.conf.json')]
versions += [tomllib.loads((root/'pyproject.toml').read_text())['project']['version'],
             tomllib.loads((root/'src-tauri/Cargo.toml').read_text())['package']['version']]
lock = json.loads((root / 'package-lock.json').read_text())
versions += [lock['version'], lock['packages']['']['version']]
versions += [package['version'] for package in tomllib.loads((root / 'src-tauri/Cargo.lock').read_text())['package']
             if package['name'] == 'openvoice4u']
assert len(set(versions)) == 1, f'Versions disagree: {versions}'
if len(sys.argv) > 1:
    assert sys.argv[1] == 'v'+versions[0], 'Release tag must match application version'
runtime = root / 'release-runtime'
if runtime.exists():
    for forbidden in ('data', 'voices', 'generations', 'models', 'portable-mode', 'settings.json', 'generation_history.json'):
        assert not (runtime / forbidden).exists(), f'Private runtime content: {forbidden}'
    assert (runtime / 'python.exe').is_file()
    assert (runtime / 'dependency-inventory.json').is_file()
print('Release inputs checked:', versions[0])
