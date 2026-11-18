"""Shared orchestration for the full, SFT-only and PPO-only entry points."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from utils.training_config import ROOT, TrainingConfig, atomic_json, read_json, config_args, digest
from utils.experiment import create_run, stage_state, append_jsonl, stage_summary, link_model_file
from utils.training_console import run_logged_process


def validate_sft_source(cfg):
    source=cfg.sft_source
    if source is None:raise ValueError('PPO-only training requires --sft-run, or --resume-run for an existing PPO run')
    manifest=read_json(source/'manifest.json')
    if manifest.get('stages',{}).get('sft',{}).get('state') not in ('completed','reused'):
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


def preflight(cfg):
    if Path(sys.prefix).resolve()!=ROOT/'.venv':raise RuntimeError('Use CoLLMLight-master/.venv/bin/python')
    if cfg.training_mode not in ('sft_only','ppo_only','sft_ppo'):raise ValueError('Not a training entry mode')
    has_sft=cfg.training_mode!='ppo_only';has_ppo=cfg.training_mode!='sft_only'
    mode=cfg['DATA_PREPARATION']
    if mode not in ('reuse','generate_teacher','sample_and_generate'):raise ValueError('Unknown training.data_preparation')
    if not cfg['EXPORT_AUTO'] or not cfg['EXPORT_VALIDATE']:raise ValueError('Training entries require automatic validated full-model export')
    if cfg['EXPORT_SOURCE'] not in ('auto',cfg.selection_stage+'_best'):raise ValueError('export.source conflicts with this training entry')
    if cfg['EXPORT_OUTPUT_DIR']:raise ValueError('Training entries export within their run directory; leave output_dir null')
    if not (cfg.path('STUDENT_MODEL_PATH')/'config.json').is_file():raise FileNotFoundError('Student model missing')
    teacher_records=None
    if has_ppo and (not has_sft or mode!='sample_and_generate'):
        if not cfg['TRAFFIC_INPUT_PATH'] or not cfg.path('TRAFFIC_INPUT_PATH').is_file():raise FileNotFoundError('inputs.traffic_samples is required for PPO')
    if has_sft:
        if cfg.sft_source:raise ValueError('Use train_ppo.py to reuse SFT; this entry trains SFT itself')
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
    else:validate_sft_source(cfg)
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
    if cfg.sft_source:copies=1
    needed=weights*(copies+(1 if cfg['EXPORT_DTYPE']=='float32' else 0))+768*2**20
    if shutil.disk_usage(ROOT).free<needed:raise RuntimeError(f'Need at least {needed/2**30:.1f} GiB free before starting')
    return {'mode':cfg.training_mode,'data_preparation':mode,'parent_run':str(cfg.sft_source) if cfg.sft_source else None,
        'teacher_records_reused':teacher_records,'estimated_min_free_gib':needed/2**30,
        'stages':(['data','sft'] if has_sft else ['reuse_sft'])+(['merge','ppo'] if has_ppo else [])+['export'],'cityflow_evaluation':False}


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


def train(cfg, resume=None):
    plan=preflight(cfg)
    folder=Path(resume).resolve() if resume else create_run(cfg)
    if resume:
        frozen=read_json(folder/'config/effective.json')
        expected=dict(cfg.values)
        expected['TRAFFIC_SAMPLE_PATH']=str(folder/'data/traffic_samples/SynTrain_sample_all.json')
        expected['TEACHER_OUTPUT_PATH']=str(folder/'data/teacher_data/syn_rt_data.json')
        if digest(expected)!=digest(frozen):raise ValueError('Resume configuration changed; start train_ppo.py --sft-run for an earlier SFT source')
        if read_json(folder/'manifest.json')['state']=='completed':raise ValueError('Experiment already completed')
    cfg=TrainingConfig(cfg.env_file,config_file=cfg.config_file,run_dir=folder)
    import fcntl
    lock_stream=(folder/'.training.lock').open('a+')
    try:fcntl.flock(lock_stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock_stream.close();raise RuntimeError('This experiment already has an active training process')
    lock_stream.seek(0);lock_stream.truncate();lock_stream.write(str(os.getpid()));lock_stream.flush()
    try:
        stage_summary(cfg,'preflight',plan)
        m=read_json(folder/'manifest.json');m['state']='running';atomic_json(folder/'manifest.json',m)
        has_sft=cfg.training_mode!='ppo_only';has_ppo=cfg.training_mode!='sft_only';mode=cfg['DATA_PREPARATION']
        if has_sft and mode=='sample_and_generate':run_stage(cfg,'sample','sample_training_data')
        elif has_ppo or mode=='generate_teacher':
            target=cfg.path('TRAFFIC_SAMPLE_PATH')
            if not target.exists():shutil.copy2(cfg.path('TRAFFIC_INPUT_PATH'),target)
        if not has_sft:
            meta=validate_sft_source(cfg)
            from training.collmlight.export_best_model import file_hash
            def copy_source(a,b):
                if Path(b).exists():
                    if file_hash(a)!=file_hash(b):raise ValueError(f'Reused SFT input changed: {b}')
                    return b
                return link_model_file(a,b)
            for rel in ['models/sft_best','data/sft_data','data/teacher_data']:
                source=cfg.sft_source/rel;target=folder/rel
                if source.exists():shutil.copytree(source,target,copy_function=copy_source,dirs_exist_ok=True)
            stage_state(folder,'sft','reused',parent_run=str(cfg.sft_source),adapter_sha256=meta['adapter_sha256'])
        else:
            if mode=='reuse':
                target=cfg.path('TEACHER_OUTPUT_PATH')
                if not target.exists():shutil.copy2(cfg.path('TEACHER_INPUT_PATH'),target)
            else:run_stage(cfg,'teacher','reasoning_tuning_data_synth')
            run_stage(cfg,'prepare','prepare_sft_data')
            run_stage(cfg,'sft','run_sft')
        if has_ppo:
            run_stage(cfg,'merge','merge_sft_adapter')
            run_stage(cfg,'ppo','ppo_ft')
        run_stage(cfg,'export','export_best_model')
        m=read_json(folder/'manifest.json');m.update(state='completed',final_model=str(cfg.export_dir));atomic_json(folder/'manifest.json',m)
        print('COMPLETED:',folder,flush=True)
        return folder
    except BaseException as exc:
        m=read_json(folder/'manifest.json');m.update(state='failed',error=type(exc).__name__);atomic_json(folder/'manifest.json',m)
        raise
    finally:fcntl.flock(lock_stream,fcntl.LOCK_UN);lock_stream.close()


def main(mode):
    parser=config_args(__doc__);parser.add_argument('--dry-run',action='store_true');parser.add_argument('--resume-run')
    if mode=='ppo_only':parser.add_argument('--sft-run',type=Path)
    args=parser.parse_args()
    if args.run_dir:parser.error('Use --resume-run to resume this entry')
    source=getattr(args,'sft_run',None)
    if args.resume_run:
        if source:parser.error('Use --sft-run only for a new PPO run')
        folder=Path(args.resume_run).resolve();manifest=read_json(folder/'manifest.json')
        if manifest['training_mode']!=mode:parser.error('Resume using the same training entry as the original run')
        source=manifest.get('parent_run')
    cfg=TrainingConfig(args.env_file,config_file=args.config_file,training_mode=mode,sft_source=source)
    if args.dry_run:print(json.dumps(preflight(cfg),indent=2));return
    train(cfg,args.resume_run)
