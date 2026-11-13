"""Lightweight source identity; excludes credentials, data and downloaded models."""
import hashlib,subprocess
from pathlib import Path
from .training_config import ROOT


def snapshot():
    files=list(ROOT.glob('*.py'))
    for directory in ['training','utils','framework','models']:
        files.extend((ROOT/directory).rglob('*.py'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    result={'git_head':revision.stdout.strip() if revision.returncode==0 else None,'files':hashes,
        'note':'Working-tree file hashes are authoritative when changes are not committed.'}
    return result
