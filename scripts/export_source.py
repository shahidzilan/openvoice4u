"""Export an allowlisted source tree for GitHub, without personal runtime data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['README.md', 'LICENSE', 'CONTRIBUTING.md', 'SECURITY.md', 'THIRD_PARTY_NOTICES.md',
         '.gitignore', '.gitattributes', 'package.json', 'package-lock.json', 'pyproject.toml',
         'requirements-windows-cpu.lock.txt', 'index.html', 'tsconfig.json', 'vite.config.ts',
         'src-tauri/Cargo.toml', 'src-tauri/Cargo.lock', 'src-tauri/build.rs', 'src-tauri/tauri.conf.json',
         'docs/publishing.md', 'docs/troubleshooting.md', 'docs/validation.md', 'docs/release-review.md',
         'scripts/bundle_runtime.py', 'scripts/export_source.py', 'scripts/check_release.py']
TREES = ['backend/app', 'frontend/src', 'tests', 'src-tauri/src', 'src-tauri/capabilities', 'src-tauri/icons', '.github']
EXTENSIONS = {'.py', '.tsx', '.ts', '.css', '.json', '.rs', '.png', '.ico', '.md', '.yml', '.yaml'}


def sources(root=ROOT):
    result = {root / name for name in FILES}
    for name in TREES:
        result.update(p for p in (root / name).rglob('*') if p.is_file() and p.suffix in EXTENSIONS and '__pycache__' not in p.parts)
    for path in sorted(result):
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Missing or unsafe source: {path}')
        yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='New empty output directory')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tree = output / 'OpenVoice4U-source'
    inventory = {}
    for path in sources():
        relative = path.relative_to(ROOT)
        destination = tree / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        inventory[relative.as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
    archive = output / 'OpenVoice4U-source.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as zipped:
        for name in inventory:
            zipped.write(tree / name, 'OpenVoice4U-source/' + name)
    (output / 'source-manifest.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    (output / 'SHA256SUMS.txt').write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name + '\n')
    print(json.dumps(dict(source=str(tree), archive=str(archive), files=len(inventory), bytes=archive.stat().st_size), indent=2))


if __name__ == '__main__':
    main()
