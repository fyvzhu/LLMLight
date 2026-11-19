"""Shared orchestration for the full, SFT-only and PPO-only entry points."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from utils.training_config import ROOT, TrainingConfig, atomic_json, read_json, config_args, digest
from utils.experiment import create_run, stage_state, append_jsonl, stage_summary
from utils.training_console import run_logged_process


def validate_sft_run(cfg,source):
    if source is None:raise ValueError('PPO requires --sft-run or --resume-run pointing to this training run')
    source=Path(source)
    manifest=read_json(source/'manifest.json')
    if manifest.get('stages',{}).get('sft',{}).get('state')!='completed':
        raise ValueError('The source SFT stage must be completed; the whole source run need not be completed')
    best=source/'models/sft_best'
    meta=read_json(best/'best_model.json')
    from training.collmlight.export_best_model import file_hash, weights_fingerprint
    if file_hash(best/'adapter_model.safetensors')!=meta['adapter_sha256']:
        raise ValueError('Source SFT adapter hash mismatch')
    if meta['metric']!='eval_loss' or read_json(best/'evaluation.json')['eval_loss']!=meta['score']:
        raise ValueError('Source SFT has no consistent best validation result')
    previous=read_json(source/'config/effective.json')
    base=Path(previous['STUDENT_MODEL_PATH'])
    if not base.is_absolute():base=ROOT/base
    if base.resolve()!=cfg.path('STUDENT_MODEL_PATH') and weights_fingerprint(base)!=weights_fingerprint(cfg.path('STUDENT_MODEL_PATH')):
        raise ValueError('Source SFT base model differs from the configured student')
    return meta


def preflight(cfg,entry=None,run_dir=None):
    entry=entry or cfg.training_mode
    if Path(sys.prefix).resolve()!=ROOT/'.venv':raise RuntimeError('Use CoLLMLight-master/.venv/bin/python')
    if entry not in ('sft_only','ppo_only','sft_ppo'):raise ValueError('Not a training entry mode')
    has_sft=entry!='ppo_only';has_ppo=entry!='sft_only'
    mode=cfg['DATA_PREPARATION']
    if mode not in ('reuse','generate_teacher','sample_and_generate'):raise ValueError('Unknown training.data_preparation')
    if not cfg['EXPORT_AUTO'] or not cfg['EXPORT_VALIDATE']:raise ValueError('Training entries require automatic validated full-model export')
    selected='ppo' if has_ppo else 'sft'
    if cfg['EXPORT_SOURCE'] not in ('auto',selected+'_best'):raise ValueError('export.source conflicts with this training entry')
    if cfg['EXPORT_OUTPUT_DIR']:raise ValueError('Training entries export within their run directory; leave output_dir null')
    if not (cfg.path('STUDENT_MODEL_PATH')/'config.json').is_file():raise FileNotFoundError('Student model missing')
    teacher_records=None
    sft_done=bool(run_dir and read_json(Path(run_dir)/'manifest.json')['stages'].get('sft',{}).get('state')=='completed')
    if has_ppo and (not has_sft or mode!='sample_and_generate'):
        if not cfg.path('TRAFFIC_SAMPLE_PATH').is_file():raise FileNotFoundError('inputs.traffic_samples is required for PPO')
    if has_sft and not sft_done:
        if mode=='reuse':
            if not cfg['TEACHER_INPUT_PATH'] or not cfg.path('TEACHER_INPUT_PATH').is_file():raise FileNotFoundError('inputs.teacher_data is required for reuse')
            rows=read_json(cfg.path('TEACHER_INPUT_PATH'))
            fields={'atr_user_prompt','atr_output','ra_user_prompt','ra_output','data','complex_level'}
            if not isinstance(rows,list) or any(set(row)!=fields or 'sample_id' in row['data'] for row in rows):raise ValueError('Expected original ICLR ATR/RA teacher data')
            teacher_records=len(rows)
            if teacher_records<2:raise ValueError('SFT requires at least two teacher records')
        else:
            if mode=='generate_teacher' and (not cfg['TRAFFIC_INPUT_PATH'] or not cfg.path('TRAFFIC_INPUT_PATH').is_file()):raise FileNotFoundError('inputs.traffic_samples is required for generate_teacher')
            if cfg['TEACHER_MAX_SAMPLES']==1:raise ValueError('SFT requires at least two teacher records')
            cfg.secret()
    else:validate_sft_run(cfg,run_dir)
    if has_ppo:
        if cfg['PPO_MAX_UPDATES']!=0:raise ValueError('Use PPO num_epochs/max_samples, with max_updates=0')
        from packaging.requirements import Requirement
        for line in importlib.metadata.requires('vllm') or []:
            req=Requirement(line)
            if req.name in ('transformers','torch') and not req.specifier.contains(importlib.metadata.version(req.name),prereleases=True):raise RuntimeError(f'vLLM dependency mismatch: {req}')
        if cfg['PPO_QUANTIZATION_BITS']==4:importlib.metadata.version('bitsandbytes')
        from training.collmlight.ppo_ft import validate_batch,split_records
        validate_batch(cfg)
        if not has_sft or mode!='sample_and_generate':split_records(cfg)
    weights=sum(p.stat().st_size for p in cfg.path('STUDENT_MODEL_PATH').glob('*.safetensors'))
    copies=2 if has_ppo else 1
    if run_dir and (Path(run_dir)/'models/sft_merged/model_manifest.json').exists():copies=1
    needed=weights*(copies+(1 if cfg['EXPORT_DTYPE']=='float32' else 0))+768*2**20
    if shutil.disk_usage(ROOT).free<needed:raise RuntimeError(f'Need at least {needed/2**30:.1f} GiB free before starting')
    return {'entry':entry,'data_preparation':mode,'run_directory':str(run_dir) if run_dir else None,
        'teacher_records_reused':teacher_records,'estimated_min_free_gib':needed/2**30,
        'stages':(['data','sft'] if has_sft else [])+['export_sft']+(['ppo','export_ppo'] if has_ppo else []),'cityflow_evaluation':False}


def run_stage(cfg, name, module, extra=()):
    import time
    manifest=read_json(cfg.out/'manifest.json')
    if manifest['stages'].get(name,{}).get('state') in ('completed','reused'):return
    attempt=manifest['stages'].get(name,{}).get('attempt',0)+1
    log=cfg.out/'logs/train.log'
    stage_state(cfg.out,name,'running',attempt=attempt,log='logs/train.log')
    command=[sys.executable,'-m','training.collmlight.'+module,'--config',str(cfg.config_file),'--run-dir',str(cfg.out)]
    if cfg.env_file:command+=['--env-file',str(cfg.env_file)]
    command+=list(extra)
    env=dict(os.environ,USE_TF='0',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
    print('Stage:',name,'log:',log,flush=True)
    from utils.source_snapshot import snapshot
    before=snapshot()
    started=time.perf_counter()
    try:
        result=run_logged_process(cfg,command,f'stage={name} attempt={attempt}',env)
    except BaseException as exc:
        stage_state(cfg.out,name,'failed',attempt=attempt,error=type(exc).__name__,
                    elapsed_seconds=time.perf_counter()-started)
        print(f'Stage {name} interrupted/failed; inspect {log}',flush=True)
        raise
    elapsed=time.perf_counter()-started
    after=snapshot()
    if before['files']!=after['files']:
        append_jsonl(cfg.out/'logs/events.jsonl',{'event':'code_changed_during_stage','stage':name})
    stage_state(cfg.out,name,'completed' if result.returncode==0 else 'failed',returncode=result.returncode,log=str(log.relative_to(cfg.out)),elapsed_seconds=elapsed)
    print(f'Stage: {name} {"completed" if result.returncode==0 else "failed"} ({elapsed:.1f}s)',flush=True)
    if result.returncode:raise RuntimeError(f'{name} failed: inspect {log}')


def validate_existing_run(cfg,folder,entry,resume):
    manifest=read_json(folder/'manifest.json')
    frozen=read_json(folder/'config/effective.json')
    if digest(frozen)!=manifest['configuration_sha256']:
        raise ValueError('Immutable experiment configuration was modified')
    expected=dict(cfg.values)
    expected['TRAFFIC_SAMPLE_PATH']=str(folder/'data/traffic_samples/SynTrain_sample_all.json')
    expected['TEACHER_OUTPUT_PATH']=str(folder/'data/teacher_data/syn_rt_data.json')
    if digest(expected)!=digest(frozen):
        raise ValueError('Configuration changed; same-directory staged training requires the same configuration as SFT')
    if manifest['training_mode'] not in ('sft_only','sft_ppo'):
        raise ValueError('This directory is not a unified SFT/PPO training run')
    if entry=='ppo_only':
        validate_sft_run(cfg,folder)
        started=manifest['stages'].get('ppo',{}).get('attempt',0)>0
        if not resume and started:raise ValueError('PPO already started; continue this directory using --resume-run')
        if resume and manifest['training_mode']!='sft_ppo':
            raise ValueError('PPO has not started; use --sft-run for the first PPO invocation')
        if manifest['state']=='completed' and manifest['stages'].get('export_ppo',{}).get('state')=='completed':
            raise ValueError('PPO and its export already completed')
    else:
        if manifest['training_mode']!=entry:raise ValueError('Use the entry matching this run training_mode')
        if manifest['state']=='completed':raise ValueError('Experiment already completed')
    return manifest


def publish_model(cfg,stage):
    run_stage(cfg,'export_'+stage,'export_best_model',('--stage',stage))
    selected=TrainingConfig(cfg.env_file,config_file=cfg.config_file,run_dir=cfg.out,selection_stage=stage)
    manifest=read_json(cfg.out/'manifest.json')
    if stage=='ppo' or not manifest.get('final_model'):
        manifest['final_model']=str(selected.export_dir)
    atomic_json(cfg.out/'manifest.json',manifest)


def train(cfg,entry,run_dir=None,resume=False):
    folder=Path(run_dir).resolve() if run_dir else None
    if entry=='ppo_only' and folder is None:raise ValueError('PPO requires --sft-run or --resume-run')
    if folder:
        validate_existing_run(cfg,folder,entry,resume)
        traffic=folder/'data/traffic_samples/SynTrain_sample_all.json'
        if traffic.is_file():cfg.values['TRAFFIC_SAMPLE_PATH']=str(traffic)
    plan=preflight(cfg,entry,folder)
    if folder is None:folder=create_run(cfg)
    import fcntl
    lock_stream=(folder/'.training.lock').open('a+')
    try:fcntl.flock(lock_stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock_stream.close();raise RuntimeError('This experiment already has an active training process')
    lock_stream.seek(0);lock_stream.truncate();lock_stream.write(str(os.getpid()));lock_stream.flush()
    started=False
    try:
        # Recheck after locking; never overwrite a concurrently completed stage.
        if run_dir:validate_existing_run(cfg,folder,entry,resume)
        # Stages can also be started directly. Reject those before changing run state.
        with (folder/'.stage.lock').open('a+') as stage_lock:
            try:fcntl.flock(stage_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another stage is already writing to this experiment')
        manifest=read_json(folder/'manifest.json')
        manifest.update(state='running',active_entry=entry)
        if entry=='ppo_only':manifest['training_mode']='sft_ppo'
        manifest.pop('error',None)
        atomic_json(folder/'manifest.json',manifest)
        started=True
        cfg=TrainingConfig(cfg.env_file,config_file=cfg.config_file,run_dir=folder)
        stage_summary(cfg,'preflight',plan)
        append_jsonl(folder/'logs/events.jsonl',{'event':'training_entry','entry':entry,'resume':resume})
        has_sft=entry!='ppo_only';has_ppo=entry!='sft_only';mode=cfg['DATA_PREPARATION']
        if has_sft and mode=='sample_and_generate':run_stage(cfg,'sample','sample_training_data')
        else:
            target=cfg.path('TRAFFIC_SAMPLE_PATH')
            if not target.exists() and cfg['TRAFFIC_INPUT_PATH'] and cfg.path('TRAFFIC_INPUT_PATH').is_file():
                shutil.copy2(cfg.path('TRAFFIC_INPUT_PATH'),target)
        if has_sft:
            if mode=='reuse':
                target=cfg.path('TEACHER_OUTPUT_PATH')
                if not target.exists():shutil.copy2(cfg.path('TEACHER_INPUT_PATH'),target)
            else:run_stage(cfg,'teacher','reasoning_tuning_data_synth')
            run_stage(cfg,'prepare','prepare_sft_data')
            run_stage(cfg,'sft','run_sft')
        publish_model(cfg,'sft')
        if has_ppo:
            run_stage(cfg,'ppo','ppo_ft')
            publish_model(cfg,'ppo')
        manifest=read_json(folder/'manifest.json');manifest.update(state='completed');atomic_json(folder/'manifest.json',manifest)
        print('COMPLETED:',folder,flush=True)
        return folder
    except BaseException as exc:
        if started:
            manifest=read_json(folder/'manifest.json');manifest.update(state='failed',error=type(exc).__name__);atomic_json(folder/'manifest.json',manifest)
        raise
    finally:fcntl.flock(lock_stream,fcntl.LOCK_UN);lock_stream.close()


def main(mode):
    parser=config_args(__doc__);parser.add_argument('--dry-run',action='store_true');parser.add_argument('--resume-run',type=Path)
    if mode=='ppo_only':parser.add_argument('--sft-run',type=Path)
    args=parser.parse_args()
    if args.run_dir:parser.error('Use --resume-run to resume this entry')
    source=getattr(args,'sft_run',None)
    if source and args.resume_run:parser.error('Choose --sft-run to begin PPO or --resume-run to continue it')
    folder=source or args.resume_run
    if mode=='ppo_only' and folder is None:parser.error('Choose --sft-run or --resume-run; PPO continues within that directory')
    cfg=TrainingConfig(args.env_file,config_file=args.config_file,training_mode='sft_only' if mode=='sft_only' else 'sft_ppo')
    if args.dry_run:
        if folder:
            folder=folder.resolve();validate_existing_run(cfg,folder,mode,bool(args.resume_run))
            traffic=folder/'data/traffic_samples/SynTrain_sample_all.json'
            if traffic.is_file():cfg.values['TRAFFIC_SAMPLE_PATH']=str(traffic)
        print(json.dumps(preflight(cfg,mode,folder),indent=2));return
    train(cfg,mode,folder,bool(args.resume_run))
