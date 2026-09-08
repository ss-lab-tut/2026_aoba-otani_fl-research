"""Reconstruct the published runtime from its base commit and saved patch."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = 'bd5d1380bb6a654a9ceff124537f60f964bae94e'


def verify(testbed, output):
    audit = ROOT / 'results/simulated_fall_p1_v2/audit.json'
    expected = json.loads(audit.read_text(encoding='utf-8'))['protocol']['testbed_source_sha256']
    patch = ROOT / 'results/condition_threshold_validation.testbed.patch'
    archive = subprocess.check_output(['git', '-C', str(testbed), 'archive', BASE])
    with tempfile.TemporaryDirectory(prefix='fall-runtime-') as temporary:
        target = Path(temporary).resolve()
        with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
            for member in contents:
                destination = (target / member.name).resolve()
                if not destination.is_relative_to(target):
                    raise ValueError('archive path escapes temporary directory')
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(contents.extractfile(member).read())
                else:
                    raise ValueError('only ordinary source files are supported')
        environment = dict(os.environ, GIT_CEILING_DIRECTORIES=str(target.parent))
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=target, env=environment, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=target, env=environment, check=True)
        paths = sorted((target / 'heterosense').rglob('*.py')) + [target / 'pyproject.toml']
        actual = {p.relative_to(target).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths}
        if actual != expected:
            raise ValueError('reconstructed runtime does not match experiment')
    report = dict(status='verified_base_plus_saved_patch', base_commit=BASE,
                  patch=patch.relative_to(ROOT).as_posix(),
                  patch_sha256=hashlib.sha256(patch.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                  verified_runtime_files=len(actual), runtime_sha256=actual,
                  scope='Source reconstruction only; full training was not rerun')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {len(actual)} runtime files from base commit plus published patch.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--testbed', type=Path, default=ROOT / 'heterosense-fl-testbed')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/simulated_fall_p1_v2/runtime_reconstruction.json')
    args = parser.parse_args()
    verify(args.testbed, args.output)
