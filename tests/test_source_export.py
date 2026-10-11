import importlib.util
from pathlib import Path


def test_source_export_excludes_personal_files(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/export_source.py'
    spec = importlib.util.spec_from_file_location('source_export', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in module.FILES:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('public fixture')
    for name in ['backend/app/worker.py', '.local/reference.txt', 'OpenVoice4U/data/profile.json',
                 'scripts/compare_narration_reference.py', 'backend/app/__pycache__/secret.py', 'backend/app/.env']:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('fixture')
    exported = {p.relative_to(tmp_path).as_posix() for p in module.sources(tmp_path)}
    assert 'backend/app/worker.py' in exported
    assert not any(name.startswith(('.local/', 'OpenVoice4U/')) for name in exported)
    assert 'scripts/compare_narration_reference.py' not in exported
    assert not any('__pycache__' in name or name.endswith('.env') for name in exported)
