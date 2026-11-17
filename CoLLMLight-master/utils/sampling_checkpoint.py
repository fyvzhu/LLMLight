"""One atomic, local checkpoint for the existing FTS sampling loop."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
import zipfile

import numpy as np

from .training_config import ROOT, atomic_json, digest, read_json
from .experiment import append_jsonl, stage_state, stage_summary


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class SamplingCheckpoint:
    def __init__(self, cfg, agent, env, paths, options):
        import cityflow
        from .source_snapshot import snapshot
        self.cfg = cfg
        self.paths = paths
        self.output = cfg.path('TRAFFIC_SAMPLE_PATH')
        self.path = cfg.out / 'checkpoints/sampling/latest.ckpt'
        self.interval = cfg['SAMPLING_CHECKPOINT_INTERVAL_STEPS']
        self.identity = {
            'configuration': digest(cfg.values), 'code': snapshot()['files'],
            'python': list(sys.version_info[:3]), 'numpy': np.__version__,
            'cityflow': file_hash(cityflow.__file__),
            'roadnet': file_hash(Path(paths['PATH_TO_DATA']) / env['ROADNET_FILE']),
            'flow': file_hash(Path(paths['PATH_TO_DATA']) / env['TRAFFIC_FILE']),
            'environment': env, 'agent': agent,
        }
        # Normalize numeric dictionary keys just as JSON does on disk.
        self.identity = json.loads(json.dumps(self.identity, sort_keys=True))
        self.available = self.path.is_file()
        self.metadata = None
        if self.available:
            with zipfile.ZipFile(self.path) as archive:
                self.metadata = json.loads(archive.read('metadata.json'))
                if self.metadata['identity'] != self.identity:
                    raise ValueError('Sampling checkpoint configuration, code, environment or input files changed')
                for name, expected in self.metadata['sha256'].items():
                    checksum = hashlib.sha256()
                    with archive.open(name) as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                            checksum.update(chunk)
                    if checksum.hexdigest() != expected:
                        raise ValueError(f'Corrupt sampling checkpoint: {name}')
            self._check_logs()

    def _check_logs(self):
        for relative, size in self.metadata['log_sizes'].items():
            path = self.cfg.out / relative
            if not path.is_file() or path.stat().st_size < size:
                raise ValueError(f'Sampling log missing or shorter than checkpoint: {relative}')

    def require_resumable_output(self):
        if self.output.exists() and not self.available:
            raise FileExistsError('Existing samples have no complete sampling checkpoint; cannot resume from JSON alone. Use a new run.')
        if not self.available:
            records = self.cfg.out / 'records/sampling'
            if records.exists() and any(records.iterdir()):
                raise FileExistsError('Existing sampling logs have no checkpoint; use a new run')

    def completed_result(self):
        summary = read_json(self.cfg.out / 'manifest.json')['stages'].get('sample', {}).get('summary', {})
        completion = summary.get('completion')
        if not completion:
            return None
        if completion['identity_sha256'] != digest(self.identity) or not self.output.is_file() or file_hash(self.output) != completion['samples_sha256']:
            raise ValueError('Committed sampling output or its configuration changed')
        print('[Sampling] complete data already committed; skipping simulation', flush=True)
        return summary['metrics']

    def maybe_save(self, sampler, loop, final=False):
        if loop['next_step'] % self.interval and not final:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # All files of a generation are bundled before replacing the old file.
        # A terminated writer can only leave this single disposable staging dir.
        staging = self.path.parent / '.writing'
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            engine = sampler.env.eng
            engine.snapshot().dump(str(staging / 'engine.json'))
            state = {key: value for key, value in vars(sampler).items() if key not in ('eng', 'archive')}
            environment = sampler.env
            references = [(environment, environment.eng)] + [(inter, inter.eng) for inter in environment.list_intersection]
            try:
                for obj, _ in references:
                    obj.eng = None
                with (staging / 'python.pkl').open('wb') as stream:
                    pickle.dump({'sampler': state, 'loop': loop, 'random': random.getstate(),
                                 'numpy_random': np.random.get_state()}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            finally:
                for obj, value in references:
                    obj.eng = value
            log_sizes = {}
            for path in sorted((self.cfg.out / 'records/sampling').glob('*')):
                if path.is_file():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
                    log_sizes[str(path.relative_to(self.cfg.out))] = path.stat().st_size
            metadata = {'identity': self.identity, 'next_step': loop['next_step'],
                        'simulation_time': loop['current_time'], 'records': len(sampler.FT_data),
                        'log_sizes': log_sizes,
                        'sha256': {name: file_hash(staging / name) for name in ('engine.json', 'python.pkl')}}
            with zipfile.ZipFile(staging / 'next.ckpt', 'w', compression=zipfile.ZIP_STORED) as archive:
                for name in ('engine.json', 'python.pkl'):
                    archive.write(staging / name, name)
                archive.writestr('metadata.json', json.dumps(metadata, sort_keys=True))
            with (staging / 'next.ckpt').open('rb') as stream:
                os.fsync(stream.fileno())
            # Re-read CRCs before publication; never remove the previous file first.
            with zipfile.ZipFile(staging / 'next.ckpt') as archive:
                if archive.testzip() is not None:
                    raise ValueError('New sampling checkpoint failed checksum validation')
            os.replace(staging / 'next.ckpt', self.path)
            sync_directory(self.path.parent)
            self.metadata = metadata
            self.available = True
            print(f'[Sampling checkpoint] saved step={loop["next_step"]} time={loop["current_time"]:.0f}s records={len(sampler.FT_data)} → {self.path}', flush=True)
            append_jsonl(self.cfg.out / 'logs/events.jsonl', {'event': 'sampling_checkpoint_saved',
                         'completed_steps': loop['next_step'], 'records': len(sampler.FT_data)})
        finally:
            import shutil
            shutil.rmtree(staging, ignore_errors=True)

    def restore(self, sampler):
        self._check_logs()
        engine = sampler.env.eng
        with zipfile.ZipFile(self.path) as archive, tempfile.TemporaryDirectory(dir=self.path.parent) as directory:
            native = Path(directory) / 'engine.json'
            with native.open('wb') as stream, archive.open('engine.json') as source:
                import shutil
                shutil.copyfileobj(source, stream)
            # Only checkpoints generated by this trusted local run are accepted.
            with archive.open('python.pkl') as stream:
                saved = pickle.load(stream)
            engine.load_from_file(str(native))
        sampler.__dict__.clear()
        sampler.__dict__.update(saved['sampler'])
        sampler.eng = engine
        sampler.env.eng = engine
        for inter in sampler.env.list_intersection:
            inter.eng = engine
        if abs(engine.get_current_time() - saved['loop']['current_time']) > 1e-8:
            raise ValueError('Sampling engine and Python checkpoint times disagree')
        # Initialization and uncommitted branch trials may have appended records.
        for path in (self.cfg.out / 'records/sampling').glob('*'):
            if path.is_file():
                relative = str(path.relative_to(self.cfg.out))
                if relative not in self.metadata['log_sizes']:
                    path.unlink()
                else:
                    with path.open('r+b') as stream:
                        stream.truncate(self.metadata['log_sizes'][relative])
                        os.fsync(stream.fileno())
        atomic_json(self.output, sampler.FT_data)
        random.setstate(saved['random'])
        np.random.set_state(saved['numpy_random'])
        print(f'[Sampling resume] completed={saved["loop"]["next_step"]} time={saved["loop"]["current_time"]:.0f}s records={len(sampler.FT_data)}; continuing in {self.cfg.out}', flush=True)
        append_jsonl(self.cfg.out / 'logs/events.jsonl', {'event': 'sampling_resumed',
                     'completed_steps': saved['loop']['next_step'], 'records': len(sampler.FT_data)})
        return saved['loop']

    def commit(self, result):
        stage_summary(self.cfg, 'sample', {'settings': {key: value for key, value in self.cfg.values.items() if key.startswith('SAMPLING_')},
                      'seed': self.cfg['RANDOM_SEED'], 'metrics': result,
                      'completion': {'identity_sha256': digest(self.identity), 'samples_sha256': file_hash(self.output),
                                     'completed_steps': self.metadata['next_step']}})
        stage_state(self.cfg.out, 'sample', 'completed')

    def cleanup(self):
        self.path.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(self.path.parent / '.writing', ignore_errors=True)
        if self.path.parent.exists() and not any(self.path.parent.iterdir()):
            self.path.parent.rmdir()
        parent = self.path.parent.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
