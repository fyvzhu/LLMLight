"""Unique, immutable experiment identity and append-only local telemetry."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import math
import os
import re
import socket
import shutil
from contextlib import contextmanager
from .training_config import ROOT, atomic_json, digest, read_json


def json_value(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {'schema_version': 1, 'utc': datetime.now(timezone.utc).isoformat(), **row}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(json_value(item), ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush()


def metrics(cfg, name, row):
    append_jsonl(cfg.out/'metrics'/f'{name}.jsonl', row)


def create_run(cfg, now=None):
    for key in ['RUN_NAME', 'ALGORITHM']:
        if not re.fullmatch(r'[\w.-]+', cfg[key]) or cfg[key] in ('.', '..'):
            raise ValueError('Unsafe experiment name')
    now = now or datetime.now(ZoneInfo('Asia/Shanghai'))
    if now.tzinfo is None:now=now.replace(tzinfo=ZoneInfo('Asia/Shanghai'))
    folder = cfg.path('OUTPUT_ROOT')/cfg['ALGORITHM']/(cfg['ALGORITHM']+'_'+now.strftime('%Y%m%d_%H%M%S_%f'))
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(exist_ok=False)
    for d in ['config','data/traffic_samples','data/teacher_data','models','logs','metrics','records','evaluations']:
        (folder/d).mkdir(parents=True, exist_ok=True)
    values = dict(cfg.values)
    values['TRAFFIC_SAMPLE_PATH']=str(folder/'data/traffic_samples/SynTrain_sample_all.json')
    values['TEACHER_OUTPUT_PATH']=str(folder/'data/teacher_data/syn_rt_data.json')
    import importlib.metadata
    import sys
    packages={}
    for package in ['torch','transformers','trl','peft','vllm','bitsandbytes','tensorflow-cpu','protobuf','datasets']:
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    atomic_json(folder/'config/environment.json',{'python':sys.executable,'prefix':sys.prefix,'packages':packages,
        'tensorflow_protobuf_compat':str(ROOT/'.venv/compat/tensorflow28'),'gpu_visibility':os.environ.get('CUDA_VISIBLE_DEVICES')})
    from .source_snapshot import snapshot
    atomic_json(folder/'config/code.json',snapshot())
    atomic_json(folder/'config/effective.json', values)
    atomic_json(folder/'manifest.json', {'schema_version':1,'run_id':folder.name,'algorithm':cfg['ALGORITHM'],
        'training_mode':'sft_ppo' if cfg['ENABLE_PPO'] else 'sft_only','state':'created','stages':{},
        'timezone':'Asia/Shanghai','created_at':now.isoformat(),'created_utc':now.astimezone(timezone.utc).isoformat(),'host':socket.gethostname(),
        'configuration_sha256':digest(values)})
    return folder


def stage_state(folder, stage, state, **extra):
    manifest = read_json(Path(folder)/'manifest.json')
    manifest['stages'].setdefault(stage,{}).update(state=state,**extra)
    atomic_json(Path(folder)/'manifest.json',manifest)
    append_jsonl(Path(folder)/'logs/events.jsonl', {'stage':stage,'state':state,**extra})


def stage_summary(cfg, stage, values):
    manifest=read_json(cfg.out/'manifest.json')
    manifest['stages'].setdefault(stage,{})['summary']=json_value(values)
    atomic_json(cfg.out/'manifest.json',manifest)


@contextmanager
def training_log(cfg, label):
    path=cfg.out/'logs/train.log'
    with path.open('a') as stream:
        stream.write(f'\n=== {datetime.now(timezone.utc).isoformat()} {label} ===\n')
        stream.flush()
        yield stream


def clean_workspace(folder, workspace):
    root=(Path(folder)/'tmp').resolve();path=Path(workspace).resolve()
    if not path.is_relative_to(root) or path==root:
        raise ValueError('Cleanup is restricted to this run temporary workspace')
    if path.exists():shutil.rmtree(path)
    parent=path.parent
    while parent.is_relative_to(root) and parent.exists() and not any(parent.iterdir()):
        parent.rmdir();parent=parent.parent


def link_model_file(source, target):
    """Share immutable model payloads; mutable progress/selection metadata is copied."""
    name=Path(source).name
    if name in {'adapter_model.safetensors','pytorch_model.bin','adapter_config.json',
                'config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
                'special_tokens_map.json','added_tokens.json','vocab.json','merges.txt','tokenizer.model'}:
        return os.link(source,target)
    return shutil.copy2(source,target)


def create_evaluation(parent, name):
    parent = Path(parent)
    if not re.fullmatch(r'[\w.-]+',name):raise ValueError('Unsafe evaluation name')
    folder=parent/'evaluations'/(name+'_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    for d in ['logs','metrics','records']:(folder/d).mkdir()
    return folder


@contextmanager
def evaluation_logging(folder):
    """Keep console output and failure status inside each evaluation."""
    import contextlib
    import sys
    import traceback
    from .baseline_archive import Tee
    folder=Path(folder)
    atomic_json(folder/'summary.json',{'state':'running','kind':'evaluation'})
    try:
        with (folder/'logs/console.log').open('x') as stream,contextlib.redirect_stdout(Tee(sys.stdout,stream)),contextlib.redirect_stderr(Tee(sys.stderr,stream)):
            try:yield
            except BaseException:
                traceback.print_exc(file=stream);raise
        summary=read_json(folder/'summary.json');summary['state']='completed'
        atomic_json(folder/'summary.json',summary)
    except BaseException as exc:
        summary=read_json(folder/'summary.json');summary.update(state='failed',error=type(exc).__name__)
        atomic_json(folder/'summary.json',summary)
        raise


def exclusive_stage(function):
    from functools import wraps
    @wraps(function)
    def wrapped(cfg,*args,**kwargs):
        if not cfg.run_dir:return function(cfg,*args,**kwargs)
        import fcntl
        with (cfg.out/'.stage.lock').open('a+') as stream:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another stage is already writing to this experiment')
            try:return function(cfg,*args,**kwargs)
            finally:fcntl.flock(stream,fcntl.LOCK_UN)
    return wrapped
